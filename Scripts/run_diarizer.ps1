python prerequisites.py --config Config/info.yaml --b_diarize
RETURN_VALUE=$?
if [[ "$RETURN_VALUE" -eq 100 ]];
then
    echo -e 'Not all the mp3 files have been assigned the min_clusters. Please assign all to continue or switch to a testing directory'
else
    python diarizer.py --config Config/diarizer.yaml;
fi
