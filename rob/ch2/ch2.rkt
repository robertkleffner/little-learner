#lang racket
(require malt)

;; Tensor<2>
(tensor
 (tensor 7 6 2 5)
 (tensor 3 8 6 9)
 (tensor 9 4 8 5))

;; # of elements of tensor
(tlen (tensor 17 12 91 67))
(tlen (tensor (tensor 3 2 8) (tensor 7 1 9)))

;; shape : Tensor<N> -> List<Integer>
;; Find the size of each dimension of a tensor.
;; Outermost dimension is first in the list, all the way to innermost
;; dimension being last in the list.
(define shape
  (λ (t)
    (cond
      [(scalar? t) (list)]
      [else (cons (tlen t) (shape (tref t 0)))])))

(shape (tensor (tensor (tensor 5) (tensor 6) (tensor 8)) (tensor (tensor 7) (tensor 9) (tensor 5))))

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

(rank (tensor (tensor (tensor 8) (tensor 9)) (tensor (tensor 4) (tensor 7))))

;; Neat law: (rank t) == (len (shape t))