package hemingway

case class LabeledDataset(data: IndexedSeq[LabeledPoint]) {
  def split(n: Int): Seq[LabeledDataset] = {
    val sectionLen = data.length / n
    val extraLen = data.length % n

    val sectionLens = Iterator.fill(extraLen)(sectionLen + 1) ++ Iterator.fill(n - extraLen)(sectionLen)
    val divPoints = sectionLens.scanLeft(0)(_ + _).toIndexedSeq

    for (i <- 0 until n)
      yield new LabeledDataset(data.slice(divPoints(i), divPoints(i + 1)))
  }
}
