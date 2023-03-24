#lang racket
(require malt)

;; ===============================
;; PREVIOUS CHAPTER UTILITIES
;; ===============================

(declare-hyper revs)
(declare-hyper alpha)
(declare-hyper batch-size)
(declare-hyper mu)

;; data-set for a simple, linear line in 2d
(define line-xs
  (tensor 2.0 1.0 4.0 3.0))
(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

;; line : Scalar -> Tensor<2> -> Scalar
;; The standard function for a planar line.
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

;; revise : (Tensor<N> -> Tensor<N>) -> Natural -> Tensor<N>
;; Given a revision function and an initial set of parameter values, iterate
;; revs times on the parameter values using the revision function. The output
;; of revision i is used as input to revision i+1. The final result is the output
;; of the revision when revs is 0.
(define revise
  (λ (f revs theta)
    (cond
      [(zero? revs) theta]
      [else (revise f (sub1 revs) (f theta))])))

;; data set for a parabola in the 2D plane
(define quad-xs (tensor -1.0 0.0 1.0 2.0 3.0))
(define quad-ys (tensor 2.55 2.1 4.35 10.2 18.25))

;; quad : Scalar -> Tensor<3> -> Scalar
;; The general form of a quadratic function, with the parameters
;; of the quadratic function given last.
(define quad
  (λ (t)
    (λ (theta)
      (+ (* (ref theta 0) (sqr t))
         (+ (* (ref theta 1) t) (ref theta 2))))))

;; data set for a plane in 2D
(define plane-xs
  (tensor
   (tensor 1.0 2.05)
   (tensor 1.0 3.0)
   (tensor 2.0 2.0)
   (tensor 2.0 3.91)
   (tensor 3.0 6.13)
   (tensor 4.0 8.09)))
(define plane-ys
  (tensor 13.99 15.99 18.0 22.4 30.2 37.94))

;; plane : Tensor<2> -> Tensor<2> -> Scalar
(define plane
  (λ (t)
    (λ (theta)
      (+ (dot-product (ref theta 0) t) (ref theta 1)))))

;; samples : Integer, Integer -> List<Integer>
;; Creates a list of s elements in the range of 0 to n.
(define samples
  (λ (n s)
    (sampled n s (list))))

(define sampled
  (λ (n i a)
    (cond
      [(zero? i) a]
      [else
       (sampled n (sub1 i) (cons (random n) a))])))

;; when an objective function uses a fixed batch size each iteration, it's stochastic gradient descent
(define sampling-obj
  (λ (expectant xs ys)
    (let ([n (tlen xs)])
      (λ (theta)
        (let ([b (samples n batch-size)])
          ((expectant (trefs xs b) (trefs ys b)) theta))))))

;; naked-i : Tensor<*> -> Tensor<*>
(define naked-i
  (λ (p) (let ([P p]) P)))

;; naked-d : Tensor<*> -> Tensor<*>
(define naked-d
  (λ (P) (let ([p P]) p)))

;; naked-u : Tensor<*>, Scalar -> Tensor<*>
(define naked-u
  (λ (P g) (- P (* alpha g))))

;; lonely-i : Tensor<*> -> List<Tensor<*>>
(define lonely-i
  (λ (p) (list p)))

;; lonely-d : List<Tensor<*>> -> Tensor<*>
(define lonely-d
  (λ (p) (ref p 0)))

;; lonely-u : List<Tensor<*>>, Tensor<*> -> List<Tensor<*>>
(define lonely-u
  (λ (P g) (list (- (ref P 0) (* alpha g)))))

(define gradient-descent
  (λ (inflate deflate update)
    (λ (obj theta)
      (let ([f (λ (big-theta)
                 (map update big-theta (gradient-of obj (map deflate big-theta))))])
        (map deflate (revise f revs (map inflate theta)))))))

(define lonely-gradient-descent
  (gradient-descent lonely-i lonely-d lonely-u))

(define naked-gradient-descent
  (gradient-descent naked-i naked-d naked-u))

(define velocity-i
  (λ (p) (list p (zeroes p))))

(define velocity-d
  (λ (P) (ref P 0)))

(define velocity-u
  (λ (P g)
    (let ([v (- (* mu (ref P 1)) (* alpha g))])
      (list (+ (ref P 0) v) v))))

(define velocity-gradient-descent
  (gradient-descent velocity-i velocity-d velocity-u))

;; =============================
;; CURRENT CHAPTER
;; =============================

;; smooth : Scalar, Tensor<N>, Tensor<N> -> Tensor<N>
;; Blend two tensors together, one being a historical average,
;; the other the next sample, using a variable decay rate. The
;; higher the decay rate, the less a new parameter that varies
;; wildly from the average will contribute to the result.
(define smooth
  (λ (decay-rate average g)
    (+ (* decay-rate average)
       (* (- 1.0 decay-rate) g))))

(smooth 0.9 0.0 50.3)
(smooth 0.9 5.03 22.7)
(smooth 0.9 6.8 4.3)