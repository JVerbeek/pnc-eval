shopt -s nullglob
for generator in config/generators/*; do
    if [ -e "$generator" ]; then
            gname="${generator##*/}"     
            gname="${gname%.yaml}"  
    fi
    count=0
    mkdir -p experiments/$gname
    for model in config/models/*; do
        if [ -e "$model" ]; then
                mname="${model##*/}"     
                mname="${mname%.yaml}"  
        fi
    	echo -e "$count\t$model\t$generator" >> experiments/$gname/job-array-config.txt
        ((count++))
    done
done