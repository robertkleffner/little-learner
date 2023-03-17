#lang racket
(require malt)

(+ (tensor 2) (tensor 7))
(+ (tensor 5 6 7) (tensor 2 0 1))
(+ (tensor (tensor 4 6 7) (tensor 2 0 1)) (tensor (tensor 1 2 2) (tensor 6 3 1)))

(+ 4 (tensor 3 6 5))
(+ (tensor 6 9 1) (tensor (tensor 4 3 8) (tensor 7 4 7)))

(* (tensor (tensor 4 6 5) (tensor 6 9 7)) 3)

(sqrt (tensor 9 16 25))
(sqrt (tensor (tensor 49 81 16) (tensor 64 25 36)))

(define sum-1
  (λ (t)
    (summed t (sub1 (tlen t)) 0.0)))

(define summed
  (λ (t i a)
    (cond
      [(zero? i) (+ (tref t 0) a)]
      [else (summed t (sub1 i) (+ (tref t i) a))])))

(sum-1 (tensor 10 12 14))

(sum (tensor (tensor (tensor 1 2) (tensor 3 4)) (tensor (tensor 5 6) (tensor 7 8))))