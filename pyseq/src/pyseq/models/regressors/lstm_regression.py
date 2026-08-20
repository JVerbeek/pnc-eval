from ..regressors.base_regression_models import StreamingRegressionModel

import numpy as np
import torch
from torch import nn


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx=None):
        # x: (batch, seq_len, 1) -> (batch, seq_len, output_size), one prediction per timestep
        out, hx_new = self.lstm(x, hx)
        return self.head(out), hx_new


class _StreamingLSTM(StreamingRegressionModel):
    # State never crosses a series boundary: fit() and predict_series() each start from zero.
    def __init__(self, hidden_size=32, num_layers=1, dropout=0.0, epochs=100,
                 lr=1e-3, truncation_length=None, device="cpu", random_state=None,
                 verbose=False, standardize=True,
                 predictor_window_size=1, target_window_size=1, skip_length=1):
        # truncation_length: timesteps per truncated-BPTT chunk during fit. None = backprop
        # over the whole series at once.
        #
        super().__init__(
            predictor_window_size=predictor_window_size,
            target_window_size=target_window_size,
            skip_length=skip_length,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.truncation_length = truncation_length
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.standardize = standardize

        self.net = None
        self.y_mean_ = None
        self.y_std_ = None
        self.fitted_output_size_ = None

    # -- subclass hooks -----------------------------------------------------------------
    @property
    def output_size(self):
        # Width of the network's output head, and with it the width of each training target.
        # This is where target_window_size enters the network, and the two variants differ:
        # MultiOutputLSTM emits the whole target window at once, AutoRegressiveLSTM predicts one step at a time.
        raise NotImplementedError

    def _decode(self, sequence, positions):
        # Produce (n_positions, target_window_size) scaled predictions from a scaled series.
        raise NotImplementedError

    # -- shared machinery ---------------------------------------------------------------
    def _standardize(self, values):
        return (values - self.y_mean_) / self.y_std_

    def _as_series(self, y):
        return np.asarray(y, dtype=np.float32).ravel()

    def _stream_targets(self, y_scaled, positions):
        # Window k's target starts one step after its last input timestep. Gathered in a single
        # fancy-index rather than stacking one slice per position, which built an intermediate
        # list of n_windows small arrays.
        return y_scaled[positions[:, None] + np.arange(1, self.output_size + 1)]

    def fit(self, y_s):
        # y_s: list of 1D arrays, each one full series of normal (changepoint-free) data.
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        series_list = [self._as_series(y) for y in y_s]

        # Standardize on training statistics only; the same transform is re-applied at predict
        # time and inverted on the model's output. 
        all_points = np.concatenate(series_list) if series_list else np.zeros(0, dtype=np.float32)
        if self.standardize and all_points.size:
            self.y_mean_ = all_points.mean()
            self.y_std_ = all_points.std() + 1e-8
        else:
            self.y_mean_ = np.float32(0.0)
            self.y_std_ = np.float32(1.0)

        streams = []
        for y in series_list:
            positions = self.target_positions(len(y))
            if len(positions) == 0:
                continue
            y_scaled = self._standardize(y)
            streams.append((
                # Inputs only need to run up to the last position a prediction is made from.
                torch.from_numpy(y_scaled[:positions[-1] + 1]).view(1, -1, 1).to(self.device),
                # Positions live on the device as a tensor from here on: they are only ever used
                # to index tensors, so keeping them in numpy would re-convert them on every
                # chunk of every epoch.
                torch.from_numpy(positions).to(self.device),
                torch.from_numpy(self._stream_targets(y_scaled, positions)).to(self.device),
            ))

        if not streams:
            raise ValueError(
                "No series long enough to fit on: every series is shorter than "
                f"predictor_window_size + target_window_size "
                f"({self.predictor_window_size} + {self.target_window_size})."
            )

        # The head is sized from target_window_size (through output_size) here and cannot change
        # afterwards, so remember what it was fitted with
        self.fitted_output_size_ = self.output_size

        self.net = _LSTMNet(self.hidden_size, self.num_layers, self.output_size, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            # Series are independent, so their order may be shuffled; the order of timepoints
            # *within* a series is what must stay intact.
            epoch_loss, epoch_n = 0.0, 0

            for stream_idx in rng.permutation(len(streams)):
                sequence, positions, targets = streams[stream_idx]
                length = sequence.shape[1]
                chunk = self.truncation_length or length

                hx = None
                for start in range(0, length, chunk):
                    stop = min(start + chunk, length)
                    out, hx = self.net.lstm(sequence[:, start:stop], hx)

                    in_chunk = torch.nonzero((positions >= start) & (positions < stop), as_tuple=True)[0]
                    if len(in_chunk):
                        preds = self.net.head(out[0, positions[in_chunk] - start])
                        loss = loss_fn(preds, targets[in_chunk])

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item() * len(in_chunk)
                        epoch_n += len(in_chunk)

                    # Carry the state's value into the next chunk but cut the backprop graph
                    # there: truncated BPTT
                    hx = tuple(state.detach() for state in hx)

            if self.verbose:
                mean_loss = epoch_loss / epoch_n if epoch_n else float("nan")
                print(f"[{type(self).__name__}] epoch {epoch + 1}/{self.epochs} - loss: {mean_loss:.6f}")

        return self

    def predict_series(self, y):
        # One full test series in, one prediction per window position out. State starts at zero
        # and is carried to the end of this series only.
        if self.net is None:
            raise ValueError(f"{type(self).__name__} must be fitted before predict_series().")

        if self.output_size != self.fitted_output_size_:
            raise ValueError(
                f"{type(self).__name__} was fitted with an output head of width "
                f"{self.fitted_output_size_} but the current window geometry needs "
                f"{self.output_size} (target_window_size={self.target_window_size}). Refit the "
                "model after changing the window geometry."
            )

        y = self._as_series(y)
        positions = self.target_positions(len(y))
        if len(positions) == 0:
            return np.zeros((0, self.target_window_size), dtype=np.float32)

        sequence = torch.from_numpy(self._standardize(y)[:positions[-1] + 1]).view(1, -1, 1).to(self.device)

        self.net.eval()
        with torch.no_grad():
            preds_scaled = self._decode(sequence, positions)

        return preds_scaled * self.y_std_ + self.y_mean_


class MultiOutputLSTM(_StreamingLSTM):
    # Predicts the whole target window in one shot from a single output head.
    #
    # target_window_size is structural here: it is the width of the output head, so it is baked
    # into the network at fit time and each training target is a full T-step window.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multi_output = True
        self.auto_regressive = False

    @property
    def output_size(self):
        return self.target_window_size

    def _decode(self, sequence, positions):
        # The full series runs through the LSTM once, then the head is applied at every
        # position at which a prediction is due.
        out, _ = self.net.lstm(sequence)
        index = torch.from_numpy(positions).to(out.device)

        return self.net.head(out[0, index]).cpu().numpy()


class AutoRegressiveLSTM(_StreamingLSTM):
    # Predicts a single step ahead and rolls that prediction forward to cover the target window.
    #
    # target_window_size is not structural here: the head is always one step wide and training is
    # always one-step-ahead, whatever T is. T only sets how many rollout steps predict_series
    # takes, so the same fitted network can serve any horizon (a refit is still required if T
    # changes, since the StackDetector's window count depends on it).
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multi_output = False
        self.auto_regressive = True

    @property
    def output_size(self):
        return 1

    def _decode(self, sequence, positions):
        if self.target_window_size == 1:
            # Nothing to roll forward, so the cell state at each position is never needed and the
            # whole series can go through in one pass: the last layer's hidden state at step t is
            # exactly the LSTM output at t, which is what the head consumes.
            out, _ = self.net.lstm(sequence)
            index = torch.from_numpy(positions).to(out.device)

            return self.net.head(out[0, index]).cpu().numpy()

        # Otherwise the rollout needs the cell state as well, and nn.LSTM only exposes it at the
        # end of a call -- hence one call per position, walking the series in skip_length sized
        # steps. The per-position states are collected in lists and concatenated once, which is
        # cheaper than writing them into a preallocated tensor one slice at a time.
        states_h, states_c = [], []
        hx = None
        previous = -1
        # Iterating the Python ints keeps numpy scalars out of the slicing in the hot loop.
        for position in positions.tolist():
            _, hx = self.net.lstm(sequence[:, previous + 1:position + 1], hx)
            states_h.append(hx[0])
            states_c.append(hx[1])
            previous = position

        # (num_layers, n_positions, hidden_size): every position becomes a batch element, so the
        # rollout below runs as one batched step per horizon step rather than per position.
        hx = (torch.cat(states_h, dim=1), torch.cat(states_c, dim=1))

        n_positions = len(positions)
        preds = np.zeros((n_positions, self.target_window_size), dtype=np.float32)
        for step in range(self.target_window_size):
            # The last layer's hidden state is the LSTM output at the current timestep.
            predicted = self.net.head(hx[0][-1])  # (n_positions, 1), still in scaled space
            preds[:, step] = predicted[:, 0].cpu().numpy()
            if step + 1 < self.target_window_size:
                # Feed each position's own prediction back in as its next input.
                _, hx = self.net.lstm(predicted.view(n_positions, 1, 1), hx)

        return preds
