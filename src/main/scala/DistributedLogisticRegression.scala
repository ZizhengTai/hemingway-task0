package hemingway

import java.time.Duration
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{exp, log}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

class DistributedLogisticRegression(
  val numMachines: Int,
  val numClasses: Int,
  val numFeatures: Int,
  val initStepSize: Double,
  val regParam: Double,
  val stepSize: (Double, Int) => Double = { (initStepSize, _) =>
    initStepSize
  }) extends LinearClassifier {

  import DistributedLogisticRegression._

  def iterationInfo: Seq[IterationInfo] = _iterationInfo
  private[this] var _iterationInfo: ArrayBuffer[IterationInfo] = _

  /** Trains the model on the given dataset.
   *
   *  @param data training dataset
   *  @param numIterations number of map-reduce iterations
   *  @param init initial model parameters
   *  @param sc Spark context
   */
  def train(data: LabeledDataset,
            numIterations: Int,
            stopLoss: Double = Double.NegativeInfinity,
            init: Option[DenseMatrix[Double]] = None)
           (implicit sc: SparkContext): Unit = {
    _params = init getOrElse DenseMatrix.fill(numClasses, numFeatures)((Random.nextDouble - 0.5) / 1e3)
    assert(params.rows == numClasses)
    assert(params.cols == numFeatures)

    _iterationInfo = ArrayBuffer.empty

    val distData = sc.parallelize(data.shuffled.split(numMachines))

    val start = System.currentTimeMillis
    for (i <- 0 until numIterations) {
      println(s"@@@ $i")

      // Perform one map-reduce update
      val iterStart = System.currentTimeMillis
      update(i, distData)
      val iterStop = System.currentTimeMillis

      _iterationInfo += IterationInfo(
        loss(data),
        Duration.ofMillis(iterStop - iterStart),
        Duration.ofMillis(iterStop - start))

      // Stop training if specified stop loss has been achieved
      val lastIters = iterationInfo.takeRight(5) map (_.loss)
      println(s"  Loss: ${lastIters.sum / lastIters.length}")
      if (lastIters.sum / lastIters.length <= stopLoss) {
        return
      }
    }
  }

  /** Performs one iteration of map-reduce.
   *
   *  @param data training datasets
   *  @param sc Spark context
   */
  private[this] def update(iteration: Int, data: RDD[LabeledDataset])(implicit sc: SparkContext): Unit = {
    val k = numClasses
    val d = numFeatures
    val γ = stepSize(initStepSize, iteration) / numMachines  // Splash
    val λ = regParam
    val currentParams = sc.broadcast(params)

    params := new DenseMatrix(
      numClasses,
      numFeatures,
      data map { localData =>
        val regr = new LogisticRegression(k, d, γ, λ)
        regr.train(localData.shuffled, Some(currentParams.value))

        regr.params.data
      } reduce { (p1, p2) =>
        (DenseVector(p1) + DenseVector(p2)).data
      })
    params /= numMachines.toDouble
  }

  /** Computes the loss of the current model parameters against the given dataset.
   *
   *  @param data the dataset to compute loss against
   */
  def loss(data: LabeledDataset): Double = {
    def squaredNorm(m: DenseMatrix[Double]): Double = {
      val v = DenseVector(m.data)
      v dot v
    }

    var l = 0.0
    for (i <- 0 until data.length) {
      val x = DenseVector(data.features(i))
      val y = data.labels(i)

      l += -params.t(::, y).dot(x) + log(sum(exp(params * x)))
    }
    l /= data.length

    // L2 regularization
    val h = 0.5 * regParam * squaredNorm(params)

    l + h
  }
}

object DistributedLogisticRegression {
  case class IterationInfo(loss: Double, iterTime: Duration, totalTime: Duration)
}
