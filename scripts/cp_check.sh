cat $1 | while read line 
do
   rsync $line/version_0/checkpoints/*.ckpt $2 --progress
done
