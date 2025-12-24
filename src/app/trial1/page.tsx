import { redirect } from 'next/navigation';

// テスト用一号（Trial #1）のマジックリンクにリダイレクト
// 有効期限: 2026/01/15 23:59:59 JST
const MAGIC_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoibWFnaWNfbGluayIsIm5hbWUiOiJUcmlhbCAjMSIsImVtYWlsIjoidHJpYWwxQGFpY2hlY2tlcnMubmV0IiwiaXNfYWRtaW4iOnRydWUsImV4cCI6MTc2ODQ4OTE5OX0.E4cLUmSoKpA7dT5MVl69NG-Q7RqOeTcdZ2GzlMlvxEo";

export default function Trial1Page() {
  redirect(`https://api.aicheckers.net/auth/magic/${MAGIC_TOKEN}`);
}
