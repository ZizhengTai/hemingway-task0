import scala.math.{log}
import scala.util.Random

import breeze.linalg._
import breeze.numerics.{exp}

/*
object Implicits {
  implicit class ArrayLinearAlgebraOps[A](val xs: Array[A]) extends AnyVal {
    def unary_-(implicit n: Numeric[A]): Array[A] = { val ys = xs.clone; ys transform n.negate; ys }

    def +=(x: A)(implicit n: Numeric[A]): Unit = xs transform (n.plus(_, x))
    def -=(x: A)(implicit n: Numeric[A]): Unit = xs transform (n.minus(_, x))
    def *=(x: A)(implicit n: Numeric[A]): Unit = xs transform (n.times(_, x))
    def /=(x: A)(implicit f: Fractional[A]): Unit = xs transform (f.div(_, x))

    def +(x: A)(implicit n: Numeric[A]): Array[A] = { val ys = xs.clone; ys += x; ys }
    def -(x: A)(implicit n: Numeric[A]): Array[A] = { val ys = xs.clone; ys -= x; ys }
    def *(x: A)(implicit n: Numeric[A]): Array[A] = { val ys = xs.clone; ys *= x; ys }
    def /(x: A)(implicit f: Fractional[A]): Array[A] = { val ys = xs.clone; ys /= x; ys }

    def +=(ys: Seq[A])(implicit n: Numeric[A]): Unit =
      ys.iterator.zipWithIndex foreach { case (y, i) => xs(i) = n.plus(xs(i), y) }
    def -=(ys: Seq[A])(implicit n: Numeric[A]): Unit =
      ys.iterator.zipWithIndex foreach { case (y, i) => xs(i) = n.minus(xs(i), y) }
    def *=(ys: Seq[A])(implicit n: Numeric[A]): Unit =
      ys.iterator.zipWithIndex foreach { case (y, i) => xs(i) = n.times(xs(i), y) }
    def /=(ys: Seq[A])(implicit f: Fractional[A]): Unit =
      ys.iterator.zipWithIndex foreach { case (y, i) => xs(i) = f.div(xs(i), y) }

    def +(ys: Seq[A])(implicit n: Numeric[A]): Array[A] = { val zs = xs.clone; zs += ys; zs }
    def -(ys: Seq[A])(implicit n: Numeric[A]): Array[A] = { val zs = xs.clone; zs -= ys; zs }
    def *(ys: Seq[A])(implicit n: Numeric[A]): Array[A] = { val zs = xs.clone; zs *= ys; zs }
    def /(ys: Seq[A])(implicit f: Fractional[A]): Array[A] = { val zs = xs.clone; zs /= ys; zs }

    def dot(ys: IndexedSeq[A])(implicit n: Numeric[A]): A = {
      var sum = n.zero
      var i = 0
      while (i < xs.length) {
        sum = n.plus(sum, n.times(xs(i), ys(i)))
        i += 1
      }
      sum
    }

    def argmax(implicit n: Numeric[A]): Int = {
      var maxIndex = 0
      var max = xs(0)
      var i = 1
      while (i < xs.length) {
        if (n.gt(xs(i), max)) {
          maxIndex = i
          max = xs(i)
        }
        i += 1
      }
      maxIndex
    }

    def argmin(implicit n: Numeric[A]): Int = {
      var minIndex = 0
      var min = xs(0)
      var i = 1
      while (i < xs.length) {
        if (n.lt(xs(i), min)) {
          minIndex = i
          min = xs(i)
        }
        i += 1
      }
      minIndex
    }
  }
}
*/

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
