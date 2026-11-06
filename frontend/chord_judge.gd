class_name ChordJudge
extends RefCounted
## Counts consecutive replies that name the wanted chord. Pure: depends only on Reply.
## Reads reply.chord, not reply.stable: the server's vote locks in the first agreed chord, a wrong one included.

# Tuned by feel: see the bridge-game spec's live check
const CONFIRM_REPLIES: int = 3  # with 3 the conditions match the server's vote; each extra reply costs 125 ms
const MIN_CONFIDENCE: float = 0.6  # must match backend stream.MIN_CONFIDENCE
const MIN_RMS: float = 0.01  # raises the server's silence threshold; cannot go below it

var count: int = 0


## True once count reaches CONFIRM_REPLIES. Anything but a confident, loud reply of the target resets the count.
func hear(reply: Reply, target: String) -> bool:
	if target != "" and reply.chord == target and reply.confidence >= MIN_CONFIDENCE and reply.rms >= MIN_RMS:
		count += 1
	else:
		count = 0
	return count >= CONFIRM_REPLIES


func reset() -> void:
	count = 0


func progress() -> float:
	return minf(float(count) / float(CONFIRM_REPLIES), 1.0)
