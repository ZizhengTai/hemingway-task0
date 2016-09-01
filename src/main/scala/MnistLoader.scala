package hemingway

import java.io.FileInputStream

class MnistLoader(path: String) {
  val height = 28
  val width = 28

  def getImages(filename: String, train: Boolean): Array[Array[Double]] = {
    val stream = new FileInputStream(path + "/" + filename)
    val numImages = if (train) 60000 else 10000
    val images = new Array[Array[Double]](numImages)

    val magicNumber = new Array[Byte](4)
    stream.read(magicNumber)
    assert(magicNumber.deep == Array[Byte](0, 0, 8, 3).deep)
    val count = new Array[Byte](4)
    stream.read(count)
    assert(count.deep == (if (train) Array[Byte](0, 0, -22, 96).deep else Array[Byte](0, 0, 39, 16).deep))
    val imHeight = new Array[Byte](4)
    stream.read(imHeight)
    assert(imHeight.deep == Array[Byte](0, 0, 0, 28).deep)
    val imWidth = new Array[Byte](4)
    stream.read(imWidth)
    assert(imWidth.deep == Array[Byte](0, 0, 0, 28).deep)

    var i = 0
    val imageBuffer = new Array[Byte](height * width)
    while (i < numImages) {
      stream.read(imageBuffer)
      images(i) = new Array[Double](imageBuffer.length + 1)
      var j = 0
      while (j < imageBuffer.length) {
        images(i)(j) = if (imageBuffer(j) != 0) 1.0 else 0.0
        j += 1
      }
      images(i)(imageBuffer.length) = 1
      i += 1
    }
    images
  }

  def getLabels(filename: String, train: Boolean): Array[Int] = {
    val stream = new FileInputStream(path + "/" + filename)
    val numLabels = if (train) 60000 else 10000

    val magicNumber = new Array[Byte](4)
    stream.read(magicNumber)
    assert(magicNumber.deep == Array[Byte](0, 0, 8, 1).deep)
    val count = new Array[Byte](4)
    stream.read(count)
    assert(count.deep == (if (train) Array[Byte](0, 0, -22, 96).deep else Array[Byte](0, 0, 39, 16).deep))

    val labels = new Array[Byte](numLabels)
    stream.read(labels)
    labels.map(e => e & 0xFF)
  }

  val trainImages = getImages("train-images.idx3-ubyte", true)
  val trainLabels = getLabels("train-labels.idx1-ubyte", true)
  val testImages = getImages("t10k-images.idx3-ubyte", false)
  val testLabels = getLabels("t10k-labels.idx1-ubyte", false)

  val trainingSet = LabeledDataset(trainLabels, trainImages)
  val testSet = LabeledDataset(testLabels, testImages)
}
