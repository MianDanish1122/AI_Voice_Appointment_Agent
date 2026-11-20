/**
 * Booking confirmation mailer: a tiny Google Apps Script web app. It runs as YOU, so the mail is sent from your own
 * Gmail (free, ~100 mails/day, no App Password needed). The voice agent POSTs {secret, to, subject, body, ics}.
 *
 * Setup (3 minutes):
 *  1. https://script.google.com -> New project -> paste this file -> rename SECRET below to any long random text.
 *  2. Deploy -> New deployment -> type "Web app" -> Execute as: Me -> Who has access: Anyone -> Deploy -> Authorize.
 *  3. Copy the Web app URL (ends with /exec). Put into .env (and the host's env):
 *       MAIL_WEBHOOK_URL=<that url>
 *       MAIL_WEBHOOK_SECRET=<the same SECRET text>
 *  After you edit this script later, use Deploy -> Manage deployments -> Edit -> New version.
 */
var SECRET = 'CHANGE-ME-to-a-long-random-text';

function doPost(e) {
  try {
    var d = JSON.parse(e.postData.contents);
    if (!d.secret || d.secret !== SECRET) return out({ ok: false, error: 'bad secret' });
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(d.to || '')) return out({ ok: false, error: 'bad address' });
    var opts = { name: d.sender_name || 'Appointments' };
    if (d.ics) opts.attachments = [Utilities.newBlob(d.ics, 'text/calendar', 'meeting.ics')];
    MailApp.sendEmail(d.to, d.subject || 'Booking confirmed', d.body || '', opts);
    return out({ ok: true });
  } catch (err) {
    return out({ ok: false, error: String(err) });
  }
}

function doGet() { return out({ ok: true, hint: 'POST only' }); }

function out(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
