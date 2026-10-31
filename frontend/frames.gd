class_name Frames
## Pure helpers for the capture -> wire path; unit-tested in tests/test_frames.gd.


## Stereo capture frames -> mono samples.
static func mono(frames: PackedVector2Array) -> PackedFloat32Array:
	var out: PackedFloat32Array = PackedFloat32Array()
	out.resize(frames.size())
	for i: int in frames.size():
		out[i] = (frames[i].x + frames[i].y) * 0.5
	return out


## Samples -> binary WebSocket frame (Float32 little-endian on every Godot target we use).
static func pack(samples: PackedFloat32Array) -> PackedByteArray:
	return samples.to_byte_array()


static func unpack(data: PackedByteArray) -> PackedFloat32Array:
	return data.to_float32_array()


## A refused or muted microphone gives exact zeros; a live one never does.
static func has_sound(samples: PackedFloat32Array) -> bool:
	for x: float in samples:
		if x != 0.0:
			return true
	return false
