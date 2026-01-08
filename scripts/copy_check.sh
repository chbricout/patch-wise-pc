cat $1 | while read line 
do
   rsync $line/version_0/checkpoints/*.ckpt chbricout@$2:/home/chbricout/Desktop/small_projects/checkpoint_composite --progress
done
