# Sentiment Analysis using pyspark

#  Machine Learning with Spark Streaming

This is project taken as part of UE19CS322 BIG DATA course at PES University. This simulates a real world scenario where you will be required to handle an enormous amount of data for predictive modelling .

Commands to Run

python3 stream.py -f sentiment -b 300   ( -f dataset -b batch size)

$SPARK_HOME/bin/spark-submit main.py -sr 5 -t True -m MLP -te True -c True ( -sr Streaming rate -t Testing the model -m model -te Training and evaluating  -c clustering)


