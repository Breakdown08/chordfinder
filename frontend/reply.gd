class_name Reply
## One StreamReply of /ws/recognize (or StreamError), with JSON nulls as "" and 0.0.

var chord: String = ""  # "" = the window is silence
var confidence: float = 0.0
var rms: float = 0.0
var stable: String = ""  # chord several consecutive windows agree on, "" = none yet
var samples: int = 0
var error: String = ""


static func parse(text: String) -> Reply:
	var r: Reply = Reply.new()
	var data: Variant = JSON.parse_string(text)
	if not data is Dictionary:
		r.error = "не JSON-объект: %s" % text.left(80)
		return r
	var d: Dictionary = data
	r.error = _text(d.get("error"))
	r.chord = _text(d.get("chord"))
	r.stable = _text(d.get("stable"))
	r.confidence = _number(d.get("confidence"))
	r.rms = _number(d.get("rms"))
	r.samples = int(_number(d.get("samples")))
	return r


static func _text(v: Variant) -> String:
	if v is String:
		var s: String = v
		return s
	return ""


static func _number(v: Variant) -> float:
	if v is float or v is int:
		var x: float = v
		return x
	return 0.0
