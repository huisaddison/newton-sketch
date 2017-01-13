SPARK_HOME="path/to/spark/"

${SPARK_HOME}/bin/spark-submit \
  --class "com.huisaddison.newtonsketch.NewtonSketch.NewtonSketchExample" \
  --master local[4] \
  --packages org.scalanlp:breeze-viz_2.11:0.11.2 \
  target/scala-2.11/newtonsketchexample_2.11-1.0.jar

