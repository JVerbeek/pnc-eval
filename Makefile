.PHONY: config clean check_clean


check_clean:
    @echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	
clean: check_clean
	rm -r experiments

config: clean
	./utils/generate-config-txt.sh