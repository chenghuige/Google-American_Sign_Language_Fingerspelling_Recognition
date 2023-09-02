# Google-American_Sign_Language_Fingerspelling_Recognition
3rd place solution  
solution doc https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434393    

# need to setup for PATH and PYTHONPATH
export PATH=/WORKPATH/tools:/WORKPARTH/tools/bin:$PATH  
export PYTHONPATH=/WORKPATH/utils:/WORKPATH/third:$PYTHONPATH  
cd projects/kaggle/aslfr/   

# generate tfrecords and mean    
cd prepare  
sh run-all.sh  
cd ..   
# train with train&sup dataset for 400 epochs  
cd src  
sh run.sh flags/final-17layers --ep=400 --online  
# fintune with train dataset only for 10 epochs  
sh run.sh flags/final-17layers --ep=400 --finetune --online  
# convert torch to keras then to tflite   
./scripts/eval-converts.sh  final-17layers.ep-400.finetune   

