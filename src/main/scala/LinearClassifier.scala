package hemingway

import breeze.linalg.{argmax, DenseMatrix, DenseVector}

trait LinearClassifier {
  /** Returns the model parameter. */
  def params: DenseMatrix[Double] = _params
  protected[this] var _params: DenseMatrix[Double] = _

  /** Predicts the labels for the given datapoints.
   *
   *  @param xs datapoints to predict labels for
   */
  def predict(xs: Seq[DenseVector[Double]]): Seq[Int] = xs map (x => argmax(params * x))
}
