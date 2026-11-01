import AppKit
import Foundation
import Vision

// usage: ocr <image> [minConfidence]
// prints JSON array of {text, conf, x, y, w, h}  (normalized, origin bottom-left)
let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: ocr <image> [minConf]\n".data(using: .utf8)!)
    exit(2)
}
let path = args[1]
let minConf = args.count >= 3 ? Float(args[2]) ?? 0.0 : 0.0

guard let img = NSImage(contentsOfFile: path),
      let tiff = img.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let cg = rep.cgImage else {
    FileHandle.standardError.write("cannot decode \(path)\n".data(using: .utf8)!)
    exit(3)
}

let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.recognitionLanguages = ["zh-Hans", "en-US"]
req.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do {
    try handler.perform([req])
} catch {
    FileHandle.standardError.write("vision error: \(error)\n".data(using: .utf8)!)
    exit(4)
}

var out: [[String: Any]] = []
for obs in (req.results ?? []) {
    guard let cand = obs.topCandidates(1).first else { continue }
    if cand.confidence < minConf { continue }
    let bb = obs.boundingBox
    out.append([
        "text": cand.string,
        "conf": Double(cand.confidence),
        "x": Double(bb.origin.x),
        "y": Double(bb.origin.y),
        "w": Double(bb.size.width),
        "h": Double(bb.size.height),
    ])
}
let data = try JSONSerialization.data(withJSONObject: out, options: [])
print(String(data: data, encoding: .utf8)!)
