import { NextRequest, NextResponse } from "next/server";

const HF_MODEL = "legekka/AI-Anime-Image-Detector-ViT";
const HF_API_URL = `https://api-inference.huggingface.co/models/${HF_MODEL}`;

type HFLabel = "ai" | "human";

type HFResponse = {
  label: HFLabel;
  score: number;
}[];

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const formData = await request.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json(
        { error: "画像ファイルが必要です" },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "画像ファイルのみ対応しています" },
        { status: 400 }
      );
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { error: "ファイルサイズは10MB以下にしてください" },
        { status: 400 }
      );
    }

    // Convert file to buffer
    const buffer = await file.arrayBuffer();

    // Call Hugging Face Inference API
    const hfResponse = await fetch(HF_API_URL, {
      method: "POST",
      headers: {
        ...(process.env.HF_TOKEN && {
          Authorization: `Bearer ${process.env.HF_TOKEN}`,
        }),
      },
      body: buffer,
    });

    if (!hfResponse.ok) {
      const errorText = await hfResponse.text();
      console.error("HuggingFace API error:", errorText);

      // Handle model loading state
      if (hfResponse.status === 503) {
        return NextResponse.json(
          {
            error: "モデルを読み込み中です。しばらくお待ちください。",
            loading: true,
          },
          { status: 503 }
        );
      }

      return NextResponse.json(
        { error: "AI判定APIでエラーが発生しました" },
        { status: 500 }
      );
    }

    const result: HFResponse = await hfResponse.json();
    const analysisTime = (Date.now() - startTime) / 1000;

    // Parse result - HF returns array of {label, score}
    const aiResult = result.find((r) => r.label === "ai");
    const humanResult = result.find((r) => r.label === "human");

    const aiProbability = aiResult?.score ?? 0.5;
    const humanProbability = humanResult?.score ?? 0.5;

    return NextResponse.json({
      isAI: aiProbability > humanProbability,
      aiProbability,
      humanProbability,
      modelUsed: HF_MODEL,
      analysisTime,
    });
  } catch (error) {
    console.error("Detection error:", error);
    return NextResponse.json(
      { error: "判定処理中にエラーが発生しました" },
      { status: 500 }
    );
  }
}
