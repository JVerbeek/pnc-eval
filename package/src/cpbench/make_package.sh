ROOT="./"
for DIR in $(find $ROOT -type d); do
        echo $DIR
	    touch $DIR/__init__.py
done
