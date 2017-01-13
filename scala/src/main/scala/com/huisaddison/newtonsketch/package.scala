package com.huisaddison.newtonsketch

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

package object NewtonSketch {
  /* Helper functions */
  def norm(x: BDV[Double]): Double = {
    sqrt(BRZsum(BRZpow(x, 2)))
  }
  def normed(x: BDV[Double]): BDV[Double] = {
    x / sqrt(BRZsum(BRZpow(x, 2)))
  }
  def log1pExp(x: Double): Double = { /* From o.a.s.mllib.util.MLUtils */
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }
  def genGaussianArray(p: Int): Array[Double] = {
      Array.fill(p)(Random.nextGaussian())
  }
  def genGaussianVector(p: Int): Vector = {
      Vectors.dense(Array.fill(p)(Random.nextGaussian()))
  }
  def genGaussianRows(sc: SparkContext, n: Int, p: Int): RDD[IndexedRow] = {
      sc.parallelize(0 to (n - 1)).map((x: Int) => IndexedRow(x,
        genGaussianVector(p)))
  }
  def genGaussianMatrix(sc: SparkContext, n: Int, p: Int): IndexedRowMatrix = {
      val rows = sc.parallelize(0 to (n - 1)).map((x: Int) => IndexedRow(x,
        genGaussianVector(p)))
      return new IndexedRowMatrix(rows)
  }
  def genBDVMatrix(sc: SparkContext, n: Int, p: Int): RDD[(Int, BDV[Double])] = {
      sc.parallelize(0 to (n - 1)).map((x: Int) => (x,
        BDV(genGaussianArray(p))))
  }
  /* http://stackoverflow.com/questions/30169841/ */
  def matrixToRDD(sc: SparkContext, m: Matrix): RDD[Vector] = {
     val columns = m.toArray.grouped(m.numRows)
     val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
     val vectors = (rows.map(row => Vectors.dense(row.toArray)))
     sc.parallelize(vectors)
  }
  def matrixToIndexedRowRDD(sc: SparkContext, m: Matrix): RDD[IndexedRow] = {
     val columns = m.toArray.grouped(m.numRows)
     val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
     val vectors = (rows zip 1.to(rows.size)).map{
       case (row, idx) => IndexedRow(idx, Vectors.dense(row.toArray))
     }
     sc.parallelize(vectors)
  }
  /* Loss calculation */
  def LogisticLoss(A: RDD[(Int, BDV[Double])], x: BDV[Double], y: BDV[Double]): Double = {
    A.map{
      case(idx, row) => {
        val ax = row dot x
        log1pExp(ax) - ax * y(idx)
      }
    }.reduce{
      (x, y) => x + y
    }
  }
  def LogisticGradient(A: RDD[(Int, BDV[Double])], x: BDV[Double], y: BDV[Double]): BDV[Double] = {
    A.map{
      case (idx, row) => {
        val ax = row dot x
        val s = (1 / (1 + exp(-ax))) - y(idx)
        s * row
      }
    }.reduce{(x, y) => x + y}
  }
  def LaplacianWeights(A: RDD[(Int, BDV[Double])], x: BDV[Double]): RDD[(Int, Double)] = {
    A.map {
      case (idx, row) => {
        val ax = row dot x
        idx -> (1 / (exp(-ax) + 2 + exp(ax)))
      }
    }
  }
  def ScaleA(A: RDD[(Int, BDV[Double])], x: BDV[Double]): RDD[(Int, BDV[Double])] = {
    A.map {
      case (idx, row) => {
        val ax = row dot x
        idx -> ((1 / sqrt(exp(-ax) + 2 + exp(ax))) * row)
      }
    }
  }
  /* Newton Gradient */
  def NewtonGradient(g: BDV[Double], dx: BDV[Double],
      Ad: RDD[(Int, BDV[Double])]): BDV[Double] = {
    Ad.map {
      case (idx, row) => {
        val s = row dot dx
        s * row
      }
    }.reduce{
      (x, y) => x + y
    } + g
  }
  /* Newton Loss */
  def NewtonLoss(g: BDV[Double], dx: BDV[Double], Ad: RDD[(Int, BDV[Double])]): Double = {
    val squared_term: Double = Ad.map {
      case (idx, row) => {
        pow(row dot dx, 2)
      }
    }.reduce{
      (x, y) => x + y
    } * 0.5
    val gradient_term: Double = g dot dx
    squared_term + gradient_term
  }
  def NewtonSketchStep(A: RDD[(Int, BDV[Double])], x: BDV[Double], dx: BDV[Double],
  y: BDV[Double]): BDV[Double] = {
    val a: Double = 0.1
    val b: Double = 0.5
    var mu: Double = 1.0
    while (LogisticLoss(A, x, y) + mu * a * (LogisticGradient(A, x, y) dot dx)
           <
           LogisticLoss(A, x + mu * dx, y) && mu > 1e-8) {
      mu = mu * b
    }
    return x + mu * dx
  }
  def SketchedA(sc: SparkContext, Ad: RDD[(Int, BDV[Double])], m: Int): RDD[(Int, BDV[Double])] = {
    Ad.flatMap {
      case (idx, row) => {
      val s = genGaussianArray(m)
        (0 to (m-1)).map{ x => (x, s(x) / m * row) }
      }
    }.reduceByKey{
      (x, y) => x + y
    }
  }
  def NewtonSketchOptimize(sc: SparkContext, A: RDD[(Int, BDV[Double])], y: BDV[Double],
  n: Int, d: Int, m: Int, max_iter: Int, gmax_iter: Int = 300): (BDV[Double], BDV[Double]) = {
    val lr: Double = 1e-1
    val loss: BDV[Double] = BDV(Array.fill(max_iter+1)(0.0))
    // var x: BDV[Double] = BDV(genGaussianArray(d))
    var x: BDV[Double] = BDV(Array.fill(d)(0.0))
    loss(0) = LogisticLoss(A, x, y)
    for (t <- 1 to max_iter) {
      // var dx: BDV[Double] = BDV(genGaussianArray(d))
      var dx: BDV[Double] = BDV(Array.fill(d)(0.0))
      val Ad: RDD[(Int, BDV[Double])] = ScaleA(A, x)
      val B: RDD[(Int, BDV[Double])] = if (m < d) SketchedA(sc, Ad, m) else Ad
      var g: BDV[Double] = LogisticGradient(A, x, y)

      var grad_hist: BDV[Double] = BDV(Array.fill(d)(1e-8))
      for (k <- 1 to gmax_iter) {
        val h: BDV[Double] = NewtonGradient(g, dx, B)
        grad_hist += BRZpow(h, 2)
        dx -= lr * (h :/ BRZpow(grad_hist, 0.5))
      }
      x = NewtonSketchStep(A, x, dx, y)
      val l: Double = LogisticLoss(A, x, y)
      loss(t) = l
    }
    (x, loss)
  }
}
