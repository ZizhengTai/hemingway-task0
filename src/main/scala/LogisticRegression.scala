package hemingway

import scala.util.Random

import breeze.linalg.{sum, DenseMatrix, DenseVector}
import breeze.numerics.exp

class LogisticRegression(
  val numClasses: Int,
  val numFeatures: Int,
  val stepSize: Double,
  val regParam: Double) extends LinearClassifier {

  /** Trains the model on the given dataset.
   *
   *  @param data training dataset
   *  @param init initial model parameters
   */
  def train(data: LabeledDataset, init: Option[DenseMatrix[Double]] = None): Unit = {
    _params = init getOrElse DenseMatrix.fill(numClasses, numFeatures)((Random.nextDouble - 0.5) / 1e3)
    assert(params.rows == numClasses)
    assert(params.cols == numFeatures)

    val gradBuffer = DenseMatrix.zeros[Double](numClasses, numFeatures)

    (data.features, data.labels).zipped foreach { (x, y) =>
      update(DenseVector(x), y, gradBuffer)
    }
  }

  /** Performs one iteration of stochastic gradient descent.
   *
   *  @param x training datapoint
   *  @param y training label
   *  @param gradBuffer buffer to store the computed gradient matrix
   */
  private[this] def update(x: DenseVector[Double], y: Int, gradBuffer: DenseMatrix[Double]): Unit = {
    /** Computes the gradient matrix of loss w.r.t. the model parameters. */
    def computeGradient(g: DenseMatrix[Double]): Unit = {
      g := regParam * params  // L2 regularization

      g(y, ::) -= x.t

      val tmp = exp(params * x)
      tmp /= sum(tmp)
      g += tmp * x.t
    }

    // Update all model parameters
    computeGradient(gradBuffer)
    gradBuffer *= stepSize
    params -= gradBuffer
  }
}
