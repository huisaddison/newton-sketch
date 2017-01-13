package com.huisaddison.newtonsketch.NewtonSketch
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import scala.math.{exp, log, log1p, pow, sqrt}
import scala.util.Random
import org.apache.spark.rdd.{RDD}
import org.apache.spark.mllib.linalg.{Matrix, Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

import breeze.linalg.{sum => BRZsum, DenseVector => BDV, axpy, linspace}
import breeze.numerics.{pow => BRZpow, sqrt => BRZsqrt}

import breeze.plot._

object NewtonSketchExample{
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NewtonSketchExample")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

	val input = sc.textFile("data/MNIST.csv")
	val A: RDD[(Int, BDV[Double])] = input.map {
	  x => {
		val v = x.split(',').map{x => x.toDouble}
		val idx: Int = v(0).toInt
		val row: BDV[Double] = BDV(v.takeRight(784))
		(idx, row)
	  }
	}
	val yy = input.map {
	  x => {
		val v = x.split(',').map{x => x.toDouble}
		val idx: Int = v(0).toInt
		val y: Double = v(1)
		(idx, y)
	  }
	}.collect()
	val y: BDV[Double] = BDV(Array.fill(1000)(0.0))
	for (i <- 0 to 999) {
	  y(yy(i)._1) = yy(i)._2
	}
    val max_iter: Int = 20
    val (w5, l5) = NewtonSketchOptimize(sc, A, y, 1000, 784, 5, max_iter)
    val (w10, l10) = NewtonSketchOptimize(sc, A, y, 1000, 784, 10, max_iter)
    val (w25, l25) = NewtonSketchOptimize(sc, A, y, 1000, 784, 25, max_iter)
    val f = Figure()
    val p = f.subplot(0)
    val t = linspace(0.0, max_iter.toDouble, max_iter + 1)
    p += plot(t, l5, name="m=5")
    p += plot(t, l10, name="m=10")
    p += plot(t, l25, name="m=25")
    p.legend = true
    f.refresh()
    f.saveas("Loss0.png")
  }
}
