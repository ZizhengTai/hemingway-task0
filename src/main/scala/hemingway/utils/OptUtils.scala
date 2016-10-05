package hemingway.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import breeze.linalg.{NumericOps, DenseVector, SparseVector, Vector}


object OptUtils {

  // load data stored in LIBSVM format
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, numFeats: Int): RDD[LabeledPoint] = {

    // read in text file
    val data = sc.textFile(filename, numSplits).coalesce(numSplits)  // note: coalesce can result in data being sent over the network. avoid this for large datasets
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.length
    val sizes = data.mapPartitionsWithIndex{ case(i, lines) =>
      Iterator.single(i -> lines.length)
    }.collect().sortBy(_._1)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithIndex { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx){

          // parse label
          val parts = line.trim().split(' ')
          var label = -1
          if (parts(0).contains("+") || parts(0).toInt == 1)
            label = 1

          // parse features
          val featureArray = parts.slice(1, parts.length)
            .map(_.split(':')
            match { case Array(i,j) => (i.toInt-1, j.toDouble)})
          var features = new SparseVector(featureArray.map(x=>x._1), featureArray.map(x=>x._2), numFeats)

          // create classification point
          Iterator.single(LabeledPoint(label,features))
        }
        else{
          Iterator.empty
        }
      }
    }
  }


  // calculate hinge loss
  def hingeLoss(point: LabeledPoint, w: Vector[Double]) : Double = {
    val y = point.label
    val X = point.features
    Math.max(1 - y * X.dot(w),0.0)
  }


  // can be used to compute train or test error
  def computeAvgLoss(data: RDD[LabeledPoint], w: Vector[Double]) : Double = {
    val n = data.count()
    data.map(hingeLoss(_, w)).reduce(_ + _) / n
  }


  // Compute the primal objective function value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computePrimalObjective(data: RDD[LabeledPoint], w: Vector[Double], lambda: Double): Double = {
    computeAvgLoss(data, w) + (0.5 * lambda * Math.pow(breeze.linalg.norm(w), 2))
  }


  // Compute the dual objective function value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computeDualObjective(data: RDD[LabeledPoint], w: Vector[Double], alpha : RDD[Vector[Double]], lambda: Double): Double = {
    val n = data.count()
    val sumAlpha = alpha.map(x => breeze.linalg.sum(x)).reduce(_ + _)
    (-lambda / 2 * Math.pow(breeze.linalg.norm(w), 2)) + (sumAlpha / n)
  }


  // Compute the duality gap value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computeDualityGap(data: RDD[LabeledPoint], w: Vector[Double], alpha: RDD[Vector[Double]], lambda: Double): Double = {
    computePrimalObjective(data, w, lambda) - computeDualObjective(data, w, alpha, lambda)
  }


  // Compute the classification error.
  def computeClassificationError(data: RDD[LabeledPoint], w:Vector[Double]) : Double = {
    val n = data.count()
    data.map(x => if(x.features.dot(w) * x.label > 0) 0.0 else 1.0).reduce(_ + _) / n
  }


  // Print summary stats after the method has finished running (primal-dual).
  def printSummaryStatsPrimalDual(algName: String, data: RDD[LabeledPoint], w: Vector[Double], alpha: RDD[Vector[Double]], lambda: Double, testData: RDD[LabeledPoint]) = {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda)
    outString = outString + "\n Total Objective Value: " + objVal
    val dualityGap = computeDualityGap(data, w, alpha, lambda)
    outString = outString + "\n Duality Gap: " + dualityGap
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }


  // Print summary stats after the method has finished running (primal only).
  def printSummaryStats(algName: String, data: RDD[LabeledPoint], w: Vector[Double], lambda: Double, testData: RDD[LabeledPoint]) =  {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda)
    outString = outString + "\n Total Objective Value: " + objVal
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }

}
