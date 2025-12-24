import { redirect } from 'next/navigation';

// テスト用一号（Trial #1）のマジックリンクにリダイレクト
// 有効期限: 2025/01/15 23:59:59 JST
const MAGIC_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoibWFnaWNfbGluayIsIm5hbWUiOiJUcmlhbCAjMSIsImVtYWlsIjoidHJpYWwxQGFpY2hlY2tlcnMubmV0IiwiaXNfYWRtaW4iOnRydWUsImV4cCI6MTczNjk1MzE5OX0.wl0yHAGEM1PSftiovOo7-B82JzMg03SYS23eI6YT34o";

export default function Trial1Page() {
  redirect(`https://api.aicheckers.net/auth/magic/${MAGIC_TOKEN}`);
}
