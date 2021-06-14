python prerequisites.py --config Config/info.yaml --b_consolidate
RETURN_VALUE=$?
if [[ "$RETURN_VALUE" -eq 100 ]];
then
    echo -e 'Not all the titles have been assigned the gold_speaker. Please assign all to continue or switch to a testing directory'
else
    python consolidate.py --config Config/diarizer.yaml;
fi