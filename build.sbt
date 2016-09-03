name := "hemingway-task0"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "com.nrinaudo" %% "kantan.csv" % "0.1.13",
  "com.nrinaudo" %% "kantan.csv-generic" % "0.1.13",
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12"
)

resolvers += Resolver.sonatypeRepo("releases")
