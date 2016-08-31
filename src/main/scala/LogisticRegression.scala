import scala.util.Random

import breeze.linalg.{argmax, sum, DenseMatrix, DenseVector}
import breeze.numerics.{exp, log}

class LogisticRegression(
  val numClasses: Int,
  val stepSize: Double,
  val regularizationFactor: Double
) {
  /** Returns the model parameters. */
  def params: DenseMatrix[Double] = _params
  private[this] var _params: DenseMatrix[Double] = _

  /** Trains the model on the given dataset.
   *
   *  @param xs training datapoints
   *  @param ys training labels
   *  @param maxIterations maximum number of iterations
   */
  def train(xs: IndexedSeq[DenseVector[Double]], ys: IndexedSeq[Int], maxIterations: Int = 100000): Unit = {
    _params = DenseMatrix.fill(numClasses, xs(0).length)((Random.nextDouble - 0.5) / 1e3)

    val gradBuffer = DenseMatrix.zeros[Double](params.rows, params.cols)

    for (i <- 0 until maxIterations) {
      // Pick one random datapoint
      val n = Random.nextInt(xs.length)
      val x = xs(n)
      val y = ys(n)

      // Update all model parameters
      update(x, y, gradBuffer)

      if (i % 1000 == 0) {
        println(i)
      }
    }
  }

  /** Predicts the label for the given datapoint.
   *
   *  @param x datapoint to predict label for
   */
  def predict(x: DenseVector[Double]): Int = argmax(params * x)

  /** Performs one iteration of stochastic gradient descent.
   *
   *  @param x training datapoint
   *  @param y training label
   *  @param gradBuffer buffer to store the computed gradient matrix
   */
  private[this] def update(x: DenseVector[Double], y: Int, gradBuffer: DenseMatrix[Double]): Unit = {
    /** Computes the gradient matrix of loss w.r.t. the model parameters. */
    def computeGradient(g: DenseMatrix[Double]): Unit = {
      g := 2 * regularizationFactor * params  // L2 regularization

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

  /** Computes the loss against the given dataset.
   *
   *  @param xs datapoints
   *  @param ys labels
   */
  def loss(xs: IndexedSeq[DenseVector[Double]], ys: IndexedSeq[Int]): Double = ???
}
