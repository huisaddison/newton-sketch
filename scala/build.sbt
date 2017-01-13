name := "NewtonSketchExample"

version := "1.0"

scalaVersion := "2.11.7"

scalaHome := Some(file("/usr/local/share/scala/"))

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "2.0.0",
    "org.apache.spark" %% "spark-mllib" % "2.0.0",
    "org.scalanlp" %% "breeze" % "0.11.2",
    "org.scalanlp" %% "breeze-natives" % "0.11.2",
    "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

