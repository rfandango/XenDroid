package xendroid.compose.core

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.util.Log
import xendroid.compose.Utils
import java.io.File
import java.io.FileInputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Per-app-session log shelving. A session = one main-process lifetime; xe.log
 * accumulates across its emulator runs (--log_append) and one logcat capture
 * spans it. The NEXT session's start zips both + exit info and prunes the
 * window - shelving after the fact covers every death mode with one path.
 */
object SessionLogs {
    private const val TAG = "SessionLogs"
    private const val CAPTURE_NAME = "logcat-current.txt"
    private const val EXIT_TS_MARKER = ".exitinfo-ts"
    // Shelve only the newest portion of runaway logs.
    private const val MAX_SHELVED_BYTES = 64L * 1024 * 1024
    private const val MAX_TRACE_BYTES = 4L * 1024 * 1024

    fun startAppSession(context: Context) {
        val xeLog = File(Utils.get_log_file_path())
        val logsDir = File(xeLog.parentFile, "logs").apply { mkdirs() }
        val capture = File(logsDir, CAPTURE_NAME)

        killStaleLogcat()
        shelvePrevious(context, xeLog, capture, logsDir)
        prune(logsDir, readKeep(context).coerceIn(1, 16))
        startCapture(capture)
    }

    /** Shortcut launches boot :emu without the main process: ensure a capture
     *  exists; shelving stays a main-process concern. */
    fun ensureCaptureRunning() {
        if (findOwnLogcatPids().isNotEmpty()) return
        val xeLog = File(Utils.get_log_file_path())
        val logsDir = File(xeLog.parentFile, "logs").apply { mkdirs() }
        startCapture(File(logsDir, CAPTURE_NAME))
    }

    private fun readKeep(context: Context): Int = runCatching {
        val handle = xendroid.compose.settings.ConfigStore(context).openLiveSnapshot()
        try {
            handle.getInt("Logging", "log_sessions_keep", 4)
        } finally {
            handle.closeString()
        }
    }.getOrDefault(4)

    private fun findOwnLogcatPids(): List<Int> {
        val myUid = android.os.Process.myUid()
        val myPid = android.os.Process.myPid()
        return File("/proc").listFiles { f -> f.name.all { it.isDigit() } }
            ?.mapNotNull { p ->
                runCatching {
                    val pid = p.name.toInt()
                    if (pid == myPid) return@runCatching null
                    // NUL-separated argv; argv[0] is the executable.
                    val argv0 = File(p, "cmdline").readBytes()
                        .takeWhile { it != 0.toByte() }.toByteArray()
                        .toString(Charsets.UTF_8)
                    if ((argv0 == "logcat" || argv0.endsWith("/logcat")) &&
                        android.system.Os.stat(p.absolutePath).st_uid == myUid)
                        pid else null
                }.getOrNull()
            } ?: emptyList()
    }

    /** Reap leftover capture processes before touching their output file. */
    private fun killStaleLogcat() {
        findOwnLogcatPids().forEach { android.os.Process.killProcess(it) }
    }

    private fun shelvePrevious(
        context: Context, xeLog: File, capture: File, logsDir: File,
    ) {
        // Plain-file leftovers from an earlier failed zip get another chance.
        val leftovers = logsDir.listFiles { f ->
            f.name.startsWith("session_") &&
                (f.name.endsWith("-xe.log") || f.name.endsWith("-logcat.txt"))
        }?.toList() ?: emptyList()
        val sources = (listOf(xeLog to "xe.log", capture to "logcat.txt") +
            leftovers.map { it to it.name.substringAfter("session_").substringAfter('-') })
            .filter { it.first.isFile && it.first.length() > 0 }
        if (sources.isEmpty()) return

        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US)
            .format(Date(sources.maxOf { it.first.lastModified() }))
        var dest = File(logsDir, "session_$stamp.zip")
        var n = 2
        while (dest.exists()) dest = File(logsDir, "session_$stamp-${n++}.zip")

        val tmp = File(logsDir, ".session.tmp")
        runCatching {
            ZipOutputStream(tmp.outputStream().buffered()).use { zip ->
                for ((file, entryName) in sources) {
                    zip.putNextEntry(ZipEntry(entryName))
                    copyTail(file, zip, MAX_SHELVED_BYTES)
                    zip.closeEntry()
                }
                writeExitInfo(context, logsDir, zip)
            }
            if (!tmp.renameTo(dest)) error("rename failed")
            sources.forEach { it.first.delete() }
            Log.i(TAG, "Shelved previous session logs to ${dest.name}")
        }.onFailure { e ->
            Log.w(TAG, "Shelving failed; keeping raw files", e)
            tmp.delete()
            // Move the unshelved sources out of the truncation paths;
            // rename() succeeds even on a full disk.
            xeLog.takeIf { it.isFile }
                ?.renameTo(File(logsDir, "session_$stamp-xe.log"))
            capture.takeIf { it.isFile }
                ?.renameTo(File(logsDir, "session_$stamp-logcat.txt"))
        }
    }

    /** Copies at most [maxBytes] of the file tail (newest content wins). */
    private fun copyTail(file: File, zip: ZipOutputStream, maxBytes: Long) {
        FileInputStream(file).use { input ->
            val len = file.length()
            if (len > maxBytes) {
                input.skip(len - maxBytes)
                zip.write("[truncated: shelved last $maxBytes of $len bytes]\n".toByteArray())
            }
            input.copyTo(zip)
        }
    }

    /** Exit reasons newer than the last shelve (crash/ANR/OOM/...), API 30+,
     *  with the platform trace attached for crashes and ANRs. */
    private fun writeExitInfo(context: Context, logsDir: File, zip: ZipOutputStream) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        runCatching {
            val marker = File(logsDir, EXIT_TS_MARKER)
            val lastSeen = marker.takeIf { it.isFile }?.readText()?.toLongOrNull() ?: 0L
            val am = context.getSystemService(ActivityManager::class.java)
            val infos = am.getHistoricalProcessExitReasons(null, 0, 16)
                .filter { it.timestamp > lastSeen }
            if (infos.isEmpty()) return
            zip.putNextEntry(ZipEntry("exit-info.txt"))
            zip.write(infos.joinToString("\n") { it.toString() }.toByteArray())
            zip.closeEntry()
            infos.forEachIndexed { i, info ->
                if (info.reason == ApplicationExitInfo.REASON_CRASH_NATIVE ||
                    info.reason == ApplicationExitInfo.REASON_ANR) {
                    runCatching {
                        info.traceInputStream?.use { trace ->
                            val name = if (info.reason ==
                                ApplicationExitInfo.REASON_CRASH_NATIVE)
                                "exit-trace-$i.tombstone.pb" else "exit-trace-$i.txt"
                            zip.putNextEntry(ZipEntry(name))
                            var left = MAX_TRACE_BYTES
                            val buf = ByteArray(64 * 1024)
                            while (left > 0) {
                                val r = trace.read(buf, 0,
                                    minOf(buf.size.toLong(), left).toInt())
                                if (r < 0) break
                                zip.write(buf, 0, r); left -= r
                            }
                            zip.closeEntry()
                        }
                    }
                }
            }
            marker.writeText(infos.maxOf { it.timestamp }.toString())
        }
    }

    private fun prune(logsDir: File, keep: Int) {
        logsDir.listFiles { f -> f.name.startsWith("session_") && f.name.endsWith(".zip") }
            ?.sortedByDescending { it.lastModified() }
            ?.drop(keep)
            ?.forEach { it.delete() }
    }

    /** Bundles every shelved session zip plus the CURRENT run's live logs
     *  into Download/xendroid-logs-<stamp>.zip. Live files are snapshot
     *  prefixes (the emulator/capture keep appending); safe to read. */
    fun exportAll(): File? {
        val xeLog = File(Utils.get_log_file_path())
        val logsDir = File(xeLog.parentFile, "logs")
        val shelved = logsDir.listFiles { f ->
            f.name.startsWith("session_") && f.name.endsWith(".zip")
        }?.sortedBy { it.lastModified() } ?: emptyList()
        val current = listOf(
            xeLog to "current/xe.log",
            File(logsDir, CAPTURE_NAME) to "current/logcat.txt",
        ).filter { it.first.isFile && it.first.length() > 0 }
        if (shelved.isEmpty() && current.isEmpty()) return null

        val downloads = android.os.Environment.getExternalStoragePublicDirectory(
            android.os.Environment.DIRECTORY_DOWNLOADS).apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dest = File(downloads, "xendroid-logs-$stamp.zip")
        val tmp = File(downloads, ".xendroid-logs.tmp")
        return runCatching {
            ZipOutputStream(tmp.outputStream().buffered()).use { zip ->
                for (f in shelved) {
                    zip.putNextEntry(ZipEntry(f.name))
                    FileInputStream(f).use { it.copyTo(zip) }
                    zip.closeEntry()
                }
                for ((f, entryName) in current) {
                    zip.putNextEntry(ZipEntry(entryName))
                    copyTail(f, zip, MAX_SHELVED_BYTES)
                    zip.closeEntry()
                }
            }
            if (!tmp.renameTo(dest)) error("rename failed")
            Log.i(TAG, "Exported logs to $dest")
            dest
        }.onFailure {
            Log.w(TAG, "Log export failed", it)
            tmp.delete()
        }.getOrNull()
    }

    /** Unprivileged logcat sees exactly this uid's entries (both processes);
     *  it streams to disk, so a crash tail is already in the file. */
    private fun startCapture(capture: File) {
        runCatching {
            ProcessBuilder("logcat", "-b", "main,system,crash", "-v", "threadtime", "-T", "1")
                .redirectOutput(ProcessBuilder.Redirect.to(capture))
                .redirectErrorStream(true)
                .start()
        }.onFailure { Log.w(TAG, "logcat capture failed to start", it) }
    }
}
