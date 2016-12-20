## extract a frame from each folder within plotwatcher/
## this is effectively a frame from each SD card

## run this from vulpes:/mnt/trails/TRAILS_STUDY_INFO

outdir=frames_"$(date +%m%d%y)"
mkdir "$outdir"

for d in plotwatcher/*/ ;
do
	vid=$(find $d -type f -name "*.TLV" | head -1)
	# echo "${outdir}"/"$(basename $d)"_frame.png
	ffmpeg -i $vid -ss 00:05:00 -vframes 1 "$outdir"/"$(basename $d)"_frame.png
done
