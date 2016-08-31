package hemingway

case class LabeledDataset(labels: Array[Int], features: Array[Array[Double]]) {
  require(labels.length == features.length)

  val length = labels.length

  def split(n: Int): Seq[LabeledDataset] = {
    val sectionLen = length / n
    val extraLen = length % n

    val sectionLens = Iterator.fill(extraLen)(sectionLen + 1) ++ Iterator.fill(n - extraLen)(sectionLen)
    val divPoints = sectionLens.scanLeft(0)(_ + _).toIndexedSeq

    for (i <- 0 until n) yield {
      val from = divPoints(i)
      val until = divPoints(i + 1)
      LabeledDataset(labels.slice(from, until), features.slice(from, until))
    }
  }
}
