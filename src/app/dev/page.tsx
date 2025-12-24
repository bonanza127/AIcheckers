import { redirect } from 'next/navigation';

// 開発者プレビュー用のマジックリンクにリダイレクト
// 有効期限: 2025/01/15 23:59:59 JST
const MAGIC_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoibWFnaWNfbGluayIsIm5hbWUiOiJEZXZlbG9wZXIgUHJldmlldyIsImVtYWlsIjoiZGV2QGFpY2hlY2tlcnMubmV0IiwiaXNfYWRtaW4iOnRydWUsImV4cCI6MTczNjk1MzE5OX0.6isV89VwRoOC-gqJ4DNyjPwMohNIElKkxSe-L3Fuy3U";

export default function DevPage() {
  redirect(`https://api.aicheckers.net/auth/magic/${MAGIC_TOKEN}`);
}
