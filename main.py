import argparse
import json
from pyspark import SparkContext
from pyspark.sql import SQLContext,SparkSession,Row
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import Tokenizer,StopWordsRemover, HashingTF,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from preprocess import *
from models import *

parser = argparse.ArgumentParser(
    description='Build model on streaming data and test it')

parser.add_argument('--stream-rate','-sr', help='Adjust streaming rate', required=False,
                    type=int, default=5)

parser.add_argument('--test','-t',help='Test the model',required=False, 
                    type=bool,default=False)

parser.add_argument('--model','-m',help='Model to be trained or tested',required=False,
                    type=str,default='MLP')

parser.add_argument('--trainEval','-te',help="Train and validate model's performance on each batch",required=False,
                    type=bool,default=True)

parser.add_argument('--clustering','-c',help="Do clustering on data",required=False,
                    type=bool,default=False)

if __name__ == '__main__':

    args = parser.parse_args()

    streamingRate = args.stream_rate
    test_flag = args.test
    model_cmd = args.model
    eval_flag = args.trainEval
    clustering_flag = args.clustering

    sc = SparkContext.getOrCreate()
    """
    df = spark.createDataFrame(tup,schema=['tweet','Sentiment'])
    df = (df.withColumn("tweet", F.regexp_replace("tweet", r"[@#&][A-Za-z0-9-]+", " ")))
    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection
    stage_1 = StringIndexer(inputCol='Sentiment',outputCol='label')
    stage_2 = Tokenizer(inputCol="tweet", outputCol="token_text")
    stopwords = StopWordsRemover().getStopWords() + ['-']
    stage_3 = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text').setOutputCol('filtered_words')
    stage_4 = NGram().setN(2).setInputCol('filtered_words').setOutputCol('bigrams')
    stage_5 = HashingTF(inputCol="bigrams", outputCol="vector",numFeatures=8000)
    
    """
    
    spark = SparkSession(sc)
    ssc = StreamingContext(sc,streamingRate)

    ham_spam_to_num = StringIndexer(inputCol='classLabel',outputCol='label')
    tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
    stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
    count_vec = HashingTF(inputCol='stop_tokens',outputCol='h_vec',numFeatures=4096)
    idf = IDF(inputCol="h_vec", outputCol="tf_idf")
    clean_up = VectorAssembler(inputCols=['tf_idf'],outputCol='features')

    data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])

    if(not test_flag):
        model = fetchNewModel(model_cmd)
    else:
        model = fetchTrainedModel(model_cmd)

    stop_flag = [0]

    streamDS = ssc.socketTextStream('localhost',6100)

    parsed_JSON_DS = streamDS.map(lambda x : json.loads(x))

    list_JSON_DS = parsed_JSON_DS.flatMap(lambda x:x.values())

    rows_DS = list_JSON_DS.map(lambda x: Row(classLabel=x['feature0'],text=x['feature1']))

    if(not clustering_flag):

        if(not test_flag):
            if(eval_flag):
                rows_DS.foreachRDD(lambda time,rdd : preprocessAndTrainEvalModel(time,rdd,ssc,stop_flag,data_prep_pipe,model))
            else:
                rows_DS.foreachRDD(lambda time,rdd : preprocessAndTrainModel(time,rdd,ssc,stop_flag,data_prep_pipe,model))
        else:
            rows_DS.foreachRDD(lambda time,rdd : preprocessAndTestModel(time,rdd,ssc,stop_flag,data_prep_pipe,model))

    else:

        rows_DS.foreachRDD(lambda time,rdd : preprocessAndClustering(time,rdd,ssc,stop_flag,data_prep_pipe))
            
    ssc.start()#actual computation or processing on data is started here
    ssc.awaitTermination()#waits for all the compuations to finish or processing to finish