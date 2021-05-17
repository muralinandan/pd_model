# python prerequisites.py --config Config/info.yaml --b_download
# RETURN_VALUE=$?
# if [[ "$RETURN_VALUE" -eq 100 ]];
# then
#     echo -e 'FATAL ERROR in GARBAGE_COLLECTOR'
# else
#     python downloader.py --config Config/downloader.yaml;
# fi
python downloader.py --config Config/downloader.yaml;



