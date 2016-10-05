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

  def iterationLogs: Seq[IterationLog] = _iterationLogs
  private[this] var _iterationLogs: ArrayBuffer[IterationLog] = _

  /** Trains the model on the given dataset.
   *
   *  @param data training dataset
   *  @param numIterations number of map-reduce iterations
   *  @param init initial model parameters
   *  @param sc Spark context
   */
  def train(data: Seq[LabeledPoint],
            numIterations: Int,
            stopLoss: Double = Double.NegativeInfinity,
            init: Option[DenseMatrix[Double]] = None)
           (implicit sc: SparkContext): Unit = {
    _params = init getOrElse DenseMatrix.fill(numClasses, numFeatures)((Random.nextDouble - 0.5) / 1e3)
    assert(params.rows == numClasses)
    assert(params.cols == numFeatures)

    _iterationLogs = ArrayBuffer.empty

    val distData = sc.parallelize(breeze.linalg.shuffle(data.toArray), numMachines).cache()

    val start = System.currentTimeMillis
    for (i <- 0 until numIterations) {
      println(s"Iteration $i")

      // Perform one map-reduce update
      val iterStart = System.currentTimeMillis
      update(i, distData)
      val iterStop = System.currentTimeMillis

      _iterationLogs += IterationLog(
        i,
        loss(data),
        Duration.ofMillis(iterStop - iterStart),
        Duration.ofMillis(iterStop - start))

      // Stop training if specified stop loss has been achieved
      val lastIters = iterationLogs.takeRight(5) map (_.loss)
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
  private[this] def update(iteration: Int, data: RDD[LabeledPoint])(implicit sc: SparkContext): Unit = {
    val k = numClasses
    val d = numFeatures
    val γ = stepSize(initStepSize, iteration) / numMachines  // Splash
    val λ = regParam
    val currentParams = sc.broadcast(params)

    params := new DenseMatrix(
      numClasses,
      numFeatures,
      data mapPartitions { (iter: Iterator[LabeledPoint]) =>
        val regr = new LogisticRegression(k, d, γ, λ)
        regr.train(iter.toIndexedSeq, Some(currentParams.value))
        Iterator.single(regr.params.data)
      } reduce { (p1, p2) =>
        (DenseVector(p1) + DenseVector(p2)).data
      })
    params /= numMachines.toDouble
  }

  /** Computes the loss of the current model parameters against the given dataset.
   *
   *  @param data the dataset to compute loss against
   */
  def loss(data: Seq[LabeledPoint]): Double = {
    def squaredNorm(m: DenseMatrix[Double]): Double = {
      val v = DenseVector(m.data)
      v dot v
    }

    var l = 0.0
    for (pt <- data) {
      val x = DenseVector(pt.features)
      val y = pt.label

      l += -params.t(::, y).dot(x) + log(sum(exp(params * x)))
    }
    l /= data.length

    // L2 regularization
    val h = 0.5 * regParam * squaredNorm(params)

    l + h
  }
}

object DistributedLogisticRegression {
  case class IterationLog(iteration: Int, loss: Double, iterTime: Duration, totalTime: Duration)
}
