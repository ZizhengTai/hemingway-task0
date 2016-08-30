import scala.util.Random

import breeze.linalg._
import breeze.numerics.{exp, log}

class LogisticRegression(
  val numClasses: Int,
  val stepSize: Double,
  val regularizationFactor: Double
) {
  /** Returns the model parameters. */
  def params: DenseMatrix[Double] = _params  //Array[Array[Double]] = _params
  private[this] var _params: DenseMatrix[Double] = _

  /** Trains the model on the given dataset.
   *
   *  @param xs training datapoints
   *  @param ys training labels
   *  @param maxIterations maximum number of iterations
   */
  def train(xs: Array[Array[Double]], ys: Array[Int], maxIterations: Int = 100000): Unit = {
    assert(xs.length == ys.length)
    val trainSize = xs.length

    _params = DenseMatrix.fill(numClasses, xs(0).length)((Random.nextDouble - 0.5) / 1e3)

    for (i <- 0 until maxIterations) {
      // Pick one random datapoint
      val n = Random.nextInt(trainSize)
      val x = DenseVector(xs(n))
      val y = ys(n)

      // Update all model parameters
      update(x, y)

      if (i % 1000 == 0) {
        println(i)
      }
    }
  }

  /** Predicts the label for the given datapoint.
   *
   *  @param x datapoint to predict label for
   */
  def predict(x: DenseVector[Double]): Int =
    argmax(DenseVector.tabulate(numClasses)(i => params.t(::, i) dot x))

  /** Performs one iteration of stochastic gradient descent.
   *
   *  @param x training datapoint
   *  @param y training label
   */
  private def update(x: DenseVector[Double], y: Int): Unit = {
    val expDots = exp(DenseVector.tabulate(numClasses)(i => params.t(::, i) dot x))
    val expDotsSum = sum(expDots)

    /** Computes the gradient of loss w.r.t. the i-th model parameter. */
    def grad(i: Int, g: DenseVector[Double]): Unit = {
      if (y == i) g := -x else g := 0.0
      g += expDots(i) / expDotsSum * x

      // L2 regularization
      g += 2 * regularizationFactor * params.t(::, i)
    }

    // Update all model parameters
    val g = DenseVector.zeros[Double](x.length)
    for (i <- 0 until params.rows) {
      grad(i, g)
      g *= stepSize

      params.t(::, i) -= g
    }
  }

  /** Computes the loss against the given dataset.
   *
   *  @param xs datapoints
   *  @param ys labels
   */
  /*
  def loss(xs: Array[Array[Double]], ys: Array[Int]): Double = {
    var l = 0.0

    for (p <- params) {
      l += p dot p
    }
    l *= regularizationFactor

    var l1 = 0.0
    for (i <- 0 until xs.length) {
      val x = xs(i)
      val y = ys(i)

      l1 -= params(y) dot x

      var tmp = 0.0
      for (p <- params) {
        tmp += exp(p dot x)
      }
      l1 += log(tmp)
    }
    l += l1 / xs.length

    l
  }*/
}
