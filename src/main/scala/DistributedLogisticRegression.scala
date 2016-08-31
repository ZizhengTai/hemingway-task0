package hemingway

import scala.util.Random

import breeze.linalg.{argmax, DenseMatrix, DenseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

class DistributedLogisticRegression(
  val numMachines: Int,
  val numClasses: Int,
  val stepSize: Double,
  val regularizationFactor: Double) extends LinearClassifier {

  /** Trains the model on the given dataset.
   *
   *  @param data training dataset
   *  @param numIterations number of map-reduce iterations
   *  @param init initial model parameters
   *  @param sc Spark context
   */
  def train(data: LabeledDataset,
            numIterations: Int,
            init: Option[DenseMatrix[Double]] = None)
           (implicit sc: SparkContext): Unit = {
    _params = init getOrElse DenseMatrix.fill(numClasses, data.features(0).length)((Random.nextDouble - 0.5) / 1e3)

    for (_ <- 0 until numIterations) {
      val distData = sc.parallelize(data.shuffle.split(numMachines))
      update(distData)
    }
  }

  /** Performs one iteration of stochastic gradient descent.
   *
   *  @param data training datasets
   *  @param sc Spark context
   */
  private[this] def update(data: RDD[LabeledDataset])(implicit sc: SparkContext): Unit = {
    val rows = params.rows
    val cols = params.cols
    val currentParams = params.data

    val K = numClasses
    val γ = stepSize
    val λ = regularizationFactor

    params := new DenseMatrix(
      params.rows,
      params.cols,
      data map { d =>
        val regr = new LogisticRegression(K, γ, λ)
        regr.train(d, Some(new DenseMatrix(rows, cols, currentParams)))

        regr.params.data
      } reduce { (p1, p2) =>
        (DenseVector(p1) + DenseVector(p2)).data
      })
    params /= numMachines.toDouble
  }
}
