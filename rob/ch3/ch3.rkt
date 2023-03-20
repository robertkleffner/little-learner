#lang racket
(require malt)

;; ===============================
;; PREVIOUS CHAPTER UTILITIES
;; ===============================

(define line-xs
  (tensor 2.0 1.0 4.0 3.0))
(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

;; x : Number
;; theta : Tensor1
(define line
  (λ (x)
    (λ (theta)
      (+ (* (ref theta 0) x) (ref theta 1)))))

;; shape : Tensor<N> -> List<Integer>
;; Find the size of each dimension of a tensor.
;; Outermost dimension is first in the list, all the way to innermost
;; dimension being last in the list.
(define shape
  (λ (t)
    (cond
      [(scalar? t) (list)]
      [else (cons (tlen t) (shape (tref t 0)))])))

;; rank : Tensor<N> -> Integer
;; Determine how many dimensions an arbitrary tensor has.
(define rank
  (λ (t)
    (ranked t 0)))

;; helper function to enable tail-call elimination with an accumulator
;; makes the recursion equivalent to a simple loop when compiled
(define ranked
  (λ (t a)
    (cond
      [(scalar? t) a]
      [else (ranked (tref t 0) (add1 a))])))

;; =============================
;; CURRENT CHAPTER
;; =============================

;; l2-loss : (DataSet -> Tensor<1> -> Tensor<1>) -> Tensor<1>, Tensor<1> -> Tensor<1> -> Scalar
;; This function, when applied to a function and a data set, is used to estimate how close a given
;; parameter set makes the function to the data set. A lower result is better, because there is less
;; 'difference' between the expected and predicted data sets.
;;    The first parameter is the target function, which we want to determine the parameters for.
;; We allow the function to be set with initial arguments that won't be changed with the parameter set.
;;    The second pair of parameters is the initial data set, with the left part being a set of inputs
;; and the right part being a corresponding set of actual results that we want the target function to
;; predict.
;;    The third parameter is a set of tensor parameters for the target function. These can be changed
;; independently of the data set to make the predicted data set approximate the actual data set.
(define l2-loss
  (λ (target)
    (λ (xs ys)
      (λ (theta)
        (let ([pred-ys ((target xs) theta)])
          (sum (sqr (- ys pred-ys))))))))

;; initial estimate of [0, 0] is quite far away
(((l2-loss line) line-xs line-ys) (list 0.0 0.0))

;; increase theta-0 a little bit, and the loss goes down a little bit!
(((l2-loss line) line-xs line-ys) (list 0.0099 0.0))

;; now we have found the rate of change ((32.59 - 33.21) / 0.0099) = -62.23
;; That is big, so we multiply it by the *learning rate* 0.01 so we don't overshoot
;; With this changed input, we are now at 5.52, much closer to ideal score
(((l2-loss line) line-xs line-ys) (list 0.6263 0.0))

;; we could keep adjusting theta-0 accordingly, which is basic successive approximation