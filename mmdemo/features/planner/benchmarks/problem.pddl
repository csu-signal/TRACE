(define (problem weights-problem)
 (:domain weights-domain)
 (:objects
   red_block blue_block green_block purple_block yellow_block - block
 )
 (:init (actual_weight red_block o_10g) (believed_weight red_block o_10g participant1) (believed_weight red_block o_10g participant2) (believed_weight red_block o_10g participant3))
 (:goal (and (believed_weight yellow_block o_50g participant1) (not (believed_weight yellow_block o_10g participant1)) (not (believed_weight yellow_block o_20g participant1)) (not (believed_weight yellow_block o_30g participant1)) (not (believed_weight yellow_block o_40g participant1)) (believed_weight yellow_block o_50g participant2) (not (believed_weight yellow_block o_10g participant2)) (not (believed_weight yellow_block o_20g participant2)) (not (believed_weight yellow_block o_30g participant2)) (not (believed_weight yellow_block o_40g participant2)) (believed_weight yellow_block o_50g participant3) (not (believed_weight yellow_block o_10g participant3)) (not (believed_weight yellow_block o_20g participant3)) (not (believed_weight yellow_block o_30g participant3)) (not (believed_weight yellow_block o_40g participant3))))
)
