(define (domain weights-domain)
 (:requirements :strips :typing :negative-preconditions :equality :conditional-effects)
 (:types block weight participant)
 (:constants
   o_20g o_50g o_40g o_10g o_30g - weight
   participant2 participant3 participant1 - participant
 )
 (:predicates (actual_weight ?block - block ?weight - weight) (believed_weight ?block - block ?weight - weight ?participant - participant) (heavier ?heavier - block ?lighter - block ?participant - participant) (is_paying_attention ?participant - participant))
 (:action set_weight
  :parameters ( ?block - block ?weight - weight)
  :precondition (and (not (actual_weight ?block o_10g)) (not (actual_weight ?block o_20g)) (not (actual_weight ?block o_30g)) (not (actual_weight ?block o_40g)) (not (actual_weight ?block o_50g)))
  :effect (and (actual_weight ?block ?weight)))
 (:action pay_attention
  :parameters ( ?participant - participant)
  :effect (and (is_paying_attention ?participant)))
 (:action stop_paying_attention
  :parameters ( ?participant - participant)
  :effect (and (not (is_paying_attention ?participant))))
 (:action compare
  :parameters ( ?left - block ?right - block)
  :precondition (and (not (= ?left ?right)))
  :effect (and (when (and (believed_weight ?right o_10g participant1) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant1)) (believed_weight ?left o_10g participant1)) (when (and (believed_weight ?left o_10g participant1) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant1)) (believed_weight ?right o_10g participant1)) (when (and (believed_weight ?right o_20g participant1) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant1)) (believed_weight ?left o_20g participant1)) (when (and (believed_weight ?left o_20g participant1) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant1)) (believed_weight ?right o_20g participant1)) (when (and (believed_weight ?right o_30g participant1) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant1)) (believed_weight ?left o_30g participant1)) (when (and (believed_weight ?left o_30g participant1) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant1)) (believed_weight ?right o_30g participant1)) (when (and (believed_weight ?right o_40g participant1) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant1)) (believed_weight ?left o_40g participant1)) (when (and (believed_weight ?left o_40g participant1) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant1)) (believed_weight ?right o_40g participant1)) (when (and (believed_weight ?right o_50g participant1) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant1)) (believed_weight ?left o_50g participant1)) (when (and (believed_weight ?left o_50g participant1) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant1)) (believed_weight ?right o_50g participant1)) (when (and (believed_weight ?right o_10g participant2) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant2)) (believed_weight ?left o_10g participant2)) (when (and (believed_weight ?left o_10g participant2) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant2)) (believed_weight ?right o_10g participant2)) (when (and (believed_weight ?right o_20g participant2) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant2)) (believed_weight ?left o_20g participant2)) (when (and (believed_weight ?left o_20g participant2) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant2)) (believed_weight ?right o_20g participant2)) (when (and (believed_weight ?right o_30g participant2) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant2)) (believed_weight ?left o_30g participant2)) (when (and (believed_weight ?left o_30g participant2) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant2)) (believed_weight ?right o_30g participant2)) (when (and (believed_weight ?right o_40g participant2) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant2)) (believed_weight ?left o_40g participant2)) (when (and (believed_weight ?left o_40g participant2) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant2)) (believed_weight ?right o_40g participant2)) (when (and (believed_weight ?right o_50g participant2) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant2)) (believed_weight ?left o_50g participant2)) (when (and (believed_weight ?left o_50g participant2) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant2)) (believed_weight ?right o_50g participant2)) (when (and (believed_weight ?right o_10g participant3) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant3)) (believed_weight ?left o_10g participant3)) (when (and (believed_weight ?left o_10g participant3) (actual_weight ?left o_10g) (actual_weight ?right o_10g) (is_paying_attention participant3)) (believed_weight ?right o_10g participant3)) (when (and (believed_weight ?right o_20g participant3) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant3)) (believed_weight ?left o_20g participant3)) (when (and (believed_weight ?left o_20g participant3) (actual_weight ?left o_20g) (actual_weight ?right o_20g) (is_paying_attention participant3)) (believed_weight ?right o_20g participant3)) (when (and (believed_weight ?right o_30g participant3) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant3)) (believed_weight ?left o_30g participant3)) (when (and (believed_weight ?left o_30g participant3) (actual_weight ?left o_30g) (actual_weight ?right o_30g) (is_paying_attention participant3)) (believed_weight ?right o_30g participant3)) (when (and (believed_weight ?right o_40g participant3) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant3)) (believed_weight ?left o_40g participant3)) (when (and (believed_weight ?left o_40g participant3) (actual_weight ?left o_40g) (actual_weight ?right o_40g) (is_paying_attention participant3)) (believed_weight ?right o_40g participant3)) (when (and (believed_weight ?right o_50g participant3) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant3)) (believed_weight ?left o_50g participant3)) (when (and (believed_weight ?left o_50g participant3) (actual_weight ?left o_50g) (actual_weight ?right o_50g) (is_paying_attention participant3)) (believed_weight ?right o_50g participant3))))
 (:action compare_one_two
  :parameters ( ?left - block ?right1 - block ?right2 - block)
  :precondition (and (not (= ?left ?right1)) (not (= ?left ?right2)) (not (= ?right1 ?right2)))
  :effect (and (when (and (believed_weight ?right1 o_10g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?left o_20g participant1)) (when (and (believed_weight ?left o_20g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right1 o_10g participant1)) (when (and (believed_weight ?left o_20g participant1) (believed_weight ?right1 o_10g participant1) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right2 o_10g participant1)) (when (and (believed_weight ?right1 o_10g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?left o_30g participant1)) (when (and (believed_weight ?left o_30g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right1 o_10g participant1)) (when (and (believed_weight ?left o_30g participant1) (believed_weight ?right1 o_10g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right2 o_20g participant1)) (when (and (believed_weight ?right1 o_20g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?left o_30g participant1)) (when (and (believed_weight ?left o_30g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right1 o_20g participant1)) (when (and (believed_weight ?left o_30g participant1) (believed_weight ?right1 o_20g participant1) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right2 o_10g participant1)) (when (and (believed_weight ?right1 o_10g participant1) (believed_weight ?right2 o_30g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?left o_40g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right2 o_30g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?right1 o_10g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right1 o_10g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?right2 o_30g participant1)) (when (and (believed_weight ?right1 o_20g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?left o_40g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right1 o_20g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right1 o_20g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right2 o_20g participant1)) (when (and (believed_weight ?right1 o_30g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?left o_40g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right1 o_30g participant1)) (when (and (believed_weight ?left o_40g participant1) (believed_weight ?right1 o_30g participant1) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right2 o_10g participant1)) (when (and (believed_weight ?right1 o_10g participant1) (believed_weight ?right2 o_40g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant1)) (believed_weight ?left o_50g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right2 o_40g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant1)) (believed_weight ?right1 o_10g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right1 o_10g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant1)) (believed_weight ?right2 o_40g participant1)) (when (and (believed_weight ?right1 o_20g participant1) (believed_weight ?right2 o_30g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?left o_50g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right2 o_30g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?right1 o_20g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right1 o_20g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant1)) (believed_weight ?right2 o_30g participant1)) (when (and (believed_weight ?right1 o_30g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?left o_50g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right2 o_20g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right1 o_30g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right1 o_30g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant1)) (believed_weight ?right2 o_20g participant1)) (when (and (believed_weight ?right1 o_40g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?left o_50g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right2 o_10g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right1 o_40g participant1)) (when (and (believed_weight ?left o_50g participant1) (believed_weight ?right1 o_40g participant1) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant1)) (believed_weight ?right2 o_10g participant1)) (when (and (believed_weight ?right1 o_10g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?left o_20g participant2)) (when (and (believed_weight ?left o_20g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right1 o_10g participant2)) (when (and (believed_weight ?left o_20g participant2) (believed_weight ?right1 o_10g participant2) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right2 o_10g participant2)) (when (and (believed_weight ?right1 o_10g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?left o_30g participant2)) (when (and (believed_weight ?left o_30g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right1 o_10g participant2)) (when (and (believed_weight ?left o_30g participant2) (believed_weight ?right1 o_10g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right2 o_20g participant2)) (when (and (believed_weight ?right1 o_20g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?left o_30g participant2)) (when (and (believed_weight ?left o_30g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right1 o_20g participant2)) (when (and (believed_weight ?left o_30g participant2) (believed_weight ?right1 o_20g participant2) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right2 o_10g participant2)) (when (and (believed_weight ?right1 o_10g participant2) (believed_weight ?right2 o_30g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?left o_40g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right2 o_30g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?right1 o_10g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right1 o_10g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?right2 o_30g participant2)) (when (and (believed_weight ?right1 o_20g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?left o_40g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right1 o_20g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right1 o_20g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right2 o_20g participant2)) (when (and (believed_weight ?right1 o_30g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?left o_40g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right1 o_30g participant2)) (when (and (believed_weight ?left o_40g participant2) (believed_weight ?right1 o_30g participant2) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right2 o_10g participant2)) (when (and (believed_weight ?right1 o_10g participant2) (believed_weight ?right2 o_40g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant2)) (believed_weight ?left o_50g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right2 o_40g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant2)) (believed_weight ?right1 o_10g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right1 o_10g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant2)) (believed_weight ?right2 o_40g participant2)) (when (and (believed_weight ?right1 o_20g participant2) (believed_weight ?right2 o_30g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?left o_50g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right2 o_30g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?right1 o_20g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right1 o_20g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant2)) (believed_weight ?right2 o_30g participant2)) (when (and (believed_weight ?right1 o_30g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?left o_50g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right2 o_20g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right1 o_30g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right1 o_30g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant2)) (believed_weight ?right2 o_20g participant2)) (when (and (believed_weight ?right1 o_40g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?left o_50g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right2 o_10g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right1 o_40g participant2)) (when (and (believed_weight ?left o_50g participant2) (believed_weight ?right1 o_40g participant2) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant2)) (believed_weight ?right2 o_10g participant2)) (when (and (believed_weight ?right1 o_10g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?left o_20g participant3)) (when (and (believed_weight ?left o_20g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right1 o_10g participant3)) (when (and (believed_weight ?left o_20g participant3) (believed_weight ?right1 o_10g participant3) (actual_weight ?left o_20g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right2 o_10g participant3)) (when (and (believed_weight ?right1 o_10g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?left o_30g participant3)) (when (and (believed_weight ?left o_30g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right1 o_10g participant3)) (when (and (believed_weight ?left o_30g participant3) (believed_weight ?right1 o_10g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right2 o_20g participant3)) (when (and (believed_weight ?right1 o_20g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?left o_30g participant3)) (when (and (believed_weight ?left o_30g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right1 o_20g participant3)) (when (and (believed_weight ?left o_30g participant3) (believed_weight ?right1 o_20g participant3) (actual_weight ?left o_30g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right2 o_10g participant3)) (when (and (believed_weight ?right1 o_10g participant3) (believed_weight ?right2 o_30g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?left o_40g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right2 o_30g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?right1 o_10g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right1 o_10g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?right2 o_30g participant3)) (when (and (believed_weight ?right1 o_20g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?left o_40g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right1 o_20g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right1 o_20g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right2 o_20g participant3)) (when (and (believed_weight ?right1 o_30g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?left o_40g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right1 o_30g participant3)) (when (and (believed_weight ?left o_40g participant3) (believed_weight ?right1 o_30g participant3) (actual_weight ?left o_40g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right2 o_10g participant3)) (when (and (believed_weight ?right1 o_10g participant3) (believed_weight ?right2 o_40g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant3)) (believed_weight ?left o_50g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right2 o_40g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant3)) (believed_weight ?right1 o_10g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right1 o_10g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_10g) (actual_weight ?right2 o_40g) (is_paying_attention participant3)) (believed_weight ?right2 o_40g participant3)) (when (and (believed_weight ?right1 o_20g participant3) (believed_weight ?right2 o_30g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?left o_50g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right2 o_30g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?right1 o_20g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right1 o_20g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_20g) (actual_weight ?right2 o_30g) (is_paying_attention participant3)) (believed_weight ?right2 o_30g participant3)) (when (and (believed_weight ?right1 o_30g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?left o_50g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right2 o_20g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right1 o_30g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right1 o_30g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_30g) (actual_weight ?right2 o_20g) (is_paying_attention participant3)) (believed_weight ?right2 o_20g participant3)) (when (and (believed_weight ?right1 o_40g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?left o_50g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right2 o_10g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right1 o_40g participant3)) (when (and (believed_weight ?left o_50g participant3) (believed_weight ?right1 o_40g participant3) (actual_weight ?left o_50g) (actual_weight ?right1 o_40g) (actual_weight ?right2 o_10g) (is_paying_attention participant3)) (believed_weight ?right2 o_10g participant3))))
)
