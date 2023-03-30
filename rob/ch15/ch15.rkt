#lang racket
(require malt)
(require malt/examples/morse)

;; ===============================
;; PREVIOUS CHAPTER UTILITIES
;; ===============================

(define epsilon 1e-08)

(declare-hyper revs)
(declare-hyper alpha)
(declare-hyper batch-size)
(declare-hyper mu)
(declare-hyper beta)

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

;; smooth : Scalar, Tensor<N>, Tensor<N> -> Tensor<N>
;; Blend two tensors together, one being a historical average,
;; the other the next sample, using a variable decay rate. The
;; higher the decay rate, the less a new parameter that varies
;; wildly from the average will contribute to the result.
(define smooth
  (λ (decay-rate average g)
    (+ (* decay-rate average)
       (* (- 1.0 decay-rate) g))))

(define rms-i
  (λ (p) (list p (zeroes p))))

(define rms-d
  (λ (P) (ref P 0)))

(define rms-u
  (λ (P g)
    (let ([r (smooth beta (ref P 1) (sqr g))])
      (let ([alpha-hat (/ alpha (+ (sqrt r) epsilon))])
        (list (- (ref P 0) (* alpha-hat g)) r)))))

(define rms-gradient-descent
  (gradient-descent rms-i rms-d rms-u))

(define adam-i
  (λ (p)
    (let ([v (zeroes p)])
      (let ([r v])
        (list p v r)))))

(define adam-d
  (λ (P) (ref P 0)))

(define adam-u
  (λ (P g)
    (let ([r (smooth beta (ref P 2) (sqr g))])
      (let ([alpha-hat (/ alpha (+ (sqrt r) epsilon))]
            [v (smooth mu (ref P 1) g)])
        (list (- (ref P 0) (* alpha-hat v)) v r)))))

(define adam-gradient-descent
  (gradient-descent adam-i adam-d adam-u))

;; rectify-0 : Scalar -> Scalar
;; If the input is less than 0, return 0.
;(define rectify-0
;  (λ (s)
;    (cond
;      [(< s 0.0) 0.0]
;      [else s])))

;; rectify : Tensor<N> -> Tensor<N>
;; If any element in the tensor is less than 0,
;; replace that element with 0.
;(define rectify
;  (ext1 rectify-0 0))

;; linear-1-1 : Tensor<1> -> (Tensor<1>, Tensor<N>) -> Tensor<N>
;; Combine the given tensor with the left parameter, then add the
;; resulting scalar to the right parameter tensor.
(define linear-1-1
  (λ (t)
    (λ (theta)
      (+ (dot-product (ref theta 0) t) (ref theta 1)))))

;; relu-1-1 : Tensor<1> -> (Tensor<1>, Tensor<N>) -> Tensor<N>
(define relu-1-1
  (λ (t)
    (λ (theta)
      (rectify ((linear-1-1 t) theta)))))

(define dot-product-2-1
  (λ (w t)
    (sum (*-2-1 w t))))

;; linear : Vector<N> -> (Matrix<M,N>, Vector<M>) -> Vector<M>
(define linear
  (λ (t)
    (λ (theta)
      (+ (dot-product-2-1 (ref theta 0) t) (ref theta 1)))))

;; relu : Vector<N> -> (Matrix<M,N>, Vector<M>) -> Vector<M>
(define relu
  (λ (t)
    (λ (theta)
      (rectify ((linear t) theta)))))

;; k-relu : Natural -> Vector<N> -> List<(Matrix<M,N>, Vector<N>)> -> Vector<N>
(define k-relu
  (λ (k)
    (λ (t)
      (λ (theta)
        (cond
          [(zero? k) t]
          [else (((k-relu
                   (sub1 k))
                  ((relu t) theta))
                 (refr theta 2))])))))

;; block : LayerFn, ((Natural, Natural), Natural) -> Block
;; Define a single layer of a neural network, via input and output size
;; and layer function.
(define block
  (λ (fn shape-lst)
    (list fn shape-lst)))

(define block-fn
  (λ (ba) (ref ba 0)))

(define block-ls
  (λ (ba) (ref ba 1)))

;; block-compose : Block, Block, Natural -> (List Tensor<N>) -> Block
(define block-compose
  (λ (f g j)
    (λ (t)
      (λ (theta)
        ((g ((f t) theta)) (refr theta j))))))

;; stack2 : Block, Block -> Block
(define stack2
  (λ (ba bb)
    (block
     (block-compose
      (block-fn ba)
      (block-fn bb)
      (len (block-ls ba)))
     (append
      (block-ls ba)
      (block-ls bb)))))

;; stack-blocks : List<Block> -> Block
(define stack-blocks
  (λ (bls) (stacked-blocks (refr bls 1) (ref bls 0))))

(define stacked-blocks
  (λ (rbls ba)
    (cond
      [(null? rbls) ba]
      [else (stacked-blocks (refr rbls 1) (stack2 ba (ref rbls 0)))])))

;; dense-block : Natural, Natural -> Block
;; Define a relu function block with input size n and output size m.
(define dense-block
  (λ (n m)
    (block relu
           (list
            (list m n)
            (list m)))))

(define init-theta
  (λ (shapes)
    (map init-shape shapes)))

(define init-shape
  (λ (s)
    (cond
      [(= (len s) 1) (zero-tensor s)]
      [(= (len s) 2) (random-tensor 0.0 (/ 2 (ref s 1)) s)]
      [(= (len s) 3) (random-tensor 0.0 (/ 2 (* (ref s 1) (ref s 2))) s)])))

(define model
  (λ (target theta)
    (λ (t)
      ((target t) theta))))

;; argmax-1 : Tensor<1> -> Natural
;; Find the index of the biggest value in the tensor.
(define argmax-1
  (λ (t)
    (let ([i (sub1 (tlen t))])
      (argmaxed t i i))))

(define argmaxed
  (λ (t i a)
    (let ([a-hat (next-a t i a)])
      (cond
        [(zero? i) a-hat]
        [else (argmaxed t (sub1 i) a-hat)]))))

(define next-a
  (λ (t i a)
    (cond
      [(> (tref t i) (tref t a)) i]
      [else a])))

;; class=-1 : Tensor<1>, Tensor<1> -> Scalar
;; Return 1.0 if two tensors represent the same class, i.e. they
;; both have the same biggest-value index. Return 0.0 otherwise.
(define class=-1
  (λ (t u)
    (cond
      [(= (argmax-1 t) (argmax-1 u)) 1.0]
      [else 0.0])))

(define class=
  (ext2 class=-1 1 1))

(define accuracy
  (λ (a-model xs ys)
    (/ (sum (class= (a-model xs) ys)) (tlen xs))))

;; ===============================
;; CURRENT CHAPTER
;; ===============================

(define corr
  (λ (t)
    (λ (theta)
      (+ (correlate (ref theta 0) t) (ref theta 1)))))

(define recu
  (λ (t)
    (λ (theta)
      (rectify ((corr t) theta)))))

(define recu-block
  (λ (b m d)
    (block recu (list (list b m d) (list b)))))

(define sum-2 sum-1)

(define sum-cols
  (ext1 sum-2 2))

(define signal-avg
  (λ (t)
    (λ (theta)
      (/ (sum-cols t) (ref (refr (shape t) (- (rank t) 2)) 0)))))

(define signal-avg-block
  (block signal-avg (list)))

(define fcn-block
  (λ (b m d)
    (stack-blocks
     (list
      (recu-block b m d)
      (recu-block b m b)))))

(define morse-fcn
  (stack-blocks
   (list
    (fcn-block 4 3 1)
    (fcn-block 8 3 4)
    (fcn-block 16 3 8)
    (fcn-block 26 3 16)
    signal-avg-block)))

(define train-morse
  (λ (network)
    (with-hypers
        ([alpha 0.0005]
         [revs 20000]
         [batch-size 8]
         [mu 0.9]
         [beta 0.999])
      (trained-morse
       (block-fn network)
       (block-ls network)))))

(define trained-morse
  (λ (classifier theta-shapes)
    (model classifier
           (adam-gradient-descent
            (sampling-obj
             (l2-loss classifier)
             morse-train-xs
             morse-train-ys)
            (init-theta theta-shapes)))))

(accuracy (train-morse morse-fcn) morse-test-xs morse-test-ys)