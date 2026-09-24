from flask import Blueprint, request, jsonify, current_app, send_file, url_for 
from database.database import get_connection 
import os 
import csv
import threading
from werkzeug.utils import secure_filename
import joblib 
import numpy as np 
import librosa 
 
from services.instrument_separation import (
    separate_audio,
    scan_existing_stems,
    find_stems_for_audio,
    SeparationInProgressError
)
 
from services.instrument_detection import ( 
    get_instrument_classes,
    detect_audio_instruments,
    save_instrument_detections,
    get_saved_instrument_detections,
    load_instrument_labels,
    extract_instrument_features
) 
from services.instrument_isolation import (
    get_or_create_isolated_instrument
) 
 
 
# ============================================================ 
# BLUEPRINT 
# ============================================================ 
 
separation_bp = Blueprint( 
    "separation", 
    __name__ 
) 
 
 
DATASET_EXTENSIONS = { 
    ".mp3", 
    ".wav", 
    ".flac", 
    ".ogg", 
    ".m4a" 
} 
 
 
# ============================================================ 
# PATHS FOR TRAINED INSTRUMENT MODEL 
# ============================================================ 
 
BASE_DIR = os.path.dirname( 
    os.path.dirname(os.path.abspath(__file__)) 
) 
 
MODEL_PATH = os.path.join( 
    BASE_DIR, 
    "models", 
    "instrument_model.joblib" 
) 
 
LABEL_PATH = os.path.join( 
    BASE_DIR, 
    "models", 
    "instrument_labels.joblib" 
) 

LABEL_CSV_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "instrument_dataset.csv"
)


# ============================================================ 
# AUDIO DURATION 
# ============================================================ 
 
def get_audio_duration(path): 
 
    if not path or not os.path.isfile(path): 
        return None 
 
    try: 
 
        import soundfile as sf 
 
        info = sf.info(path) 
 
        return round( 
            float(info.duration), 
            2 
        ) 
 
    except Exception: 
 
        try: 
 
            import wave 
 
            with wave.open(path, "rb") as w: 
 
                return round( 
                    w.getnframes() 
                    / float(w.getframerate()), 
                    2 
                ) 
 
        except Exception: 
 
            return None 
 
 
# ============================================================
# AI INSTRUMENT DETECTION ON THE SEPARATED BGM
# ============================================================

def detect_bgm_instruments(bgm_path):
    """Run the trained instrument model on the separated BGM stem."""
    if not bgm_path or not os.path.isfile(bgm_path):
        return []
    return detect_audio_instruments(bgm_path, source="BGM")


_prepare_lock = threading.Lock()
_prepared = set()


def prepare_instrument_audio(audio_id, detections):
    """Build every detected instrument's audio in the background so the
    'Play <instrument>' buttons start instantly."""
    names = [
        (d.get("instrument") or d.get("instrument_name") or "").strip()
        for d in (detections or [])
    ]
    names = [n for n in names if n]
    key = (str(audio_id), tuple(sorted(names)))
    with _prepare_lock:
        if not names or key in _prepared:
            return
        _prepared.add(key)

    def worker():
        for name in names:
            try:
                get_or_create_isolated_instrument(audio_id, name)
            except Exception as error:
                print(f"Background isolation failed for {name}: {error}")

    threading.Thread(target=worker, daemon=True).start()


def get_or_detect_bgm_instruments(audio_id, bgm_path):
    """Reuse saved BGM detections, otherwise detect on the BGM stem.

    Detections saved at upload time come from the full mix (with vocals);
    once a BGM stem exists we re-run the model on it, which is closer to
    the instrument-only audio the model was trained on.
    """
    saved = get_saved_instrument_detections(audio_id)
    if saved and any((d.get("source") or "").upper() == "BGM" for d in saved):
        return saved
    detections = detect_bgm_instruments(bgm_path)
    if detections:
        save_instrument_detections(audio_id, detections)
        return detections
    return saved


# ============================================================ 
# STEM PAYLOAD 
# ============================================================ 
 
def stem_payload( 
    path, 
    dur=None 
): 
 
    if not path or not os.path.isfile(path): 
 
        return { 
            "path": "", 
            "stream_url": "", 
            "download_url": "", 
            "duration": dur 
        } 
 
    query = { 
        "path": path 
    } 
 
    return { 
 
        "path": path, 
 
        "duration": dur, 
 
        "stream_url": url_for( 
            "separation.get_separated_audio", 
            **query 
        ), 
 
        "download_url": url_for( 
            "separation.get_separated_audio", 
            download="1", 
            **query 
        ) 
    } 
 
 
# ============================================================ 
# BUILD SEPARATION RESPONSE 
# ============================================================ 
 
def build_separation_payload( 
    audio_id, 
    audio_file_path, 
    stems_dict, 
    message="Separation ready", 
    detected_instruments=None 
): 
 
    vocals_path = stems_dict.get( 
        "vocals" 
    ) 
 
    bgm_path = ( 
        stems_dict.get("bgm") 
        or stems_dict.get("other") 
    ) 
 
    drums_path = stems_dict.get( 
        "drums" 
    ) 
 
    bass_path = stems_dict.get( 
        "bass" 
    ) 
 
    other_path = stems_dict.get( 
        "other" 
    ) 
 
    uploaded_duration = get_audio_duration( 
        audio_file_path 
    ) 
 
    vocals_duration = ( 
        get_audio_duration(vocals_path) 
        or uploaded_duration 
    ) 
 
    bgm_duration = ( 
        get_audio_duration(bgm_path) 
        or uploaded_duration 
    ) 
 
    target_duration = ( 
        uploaded_duration 
        or vocals_duration 
        or bgm_duration 
    ) 
 
    # -------------------------------------------------------- 
    # FULL SONG 
    # -------------------------------------------------------- 
 
    full_song = { 
 
        "path": audio_file_path, 
 
        "duration": target_duration, 
 
        "stream_url": url_for( 
            "audio.get_uploaded_audio", 
            path=audio_file_path 
        ), 
 
        "download_url": url_for( 
            "audio.get_uploaded_audio", 
            path=audio_file_path, 
            download="1" 
        ) 
    } 
 
    # -------------------------------------------------------- 
    # SUPPORTED INSTRUMENTS 
    # -------------------------------------------------------- 
 
    supported_instruments = ( 
        get_instrument_classes() 
    ) 

    # -------------------------------------------------------- 
    # ENRICH DETECTED INSTRUMENTS
    # -------------------------------------------------------- 

    bgm_stem = stem_payload(
        bgm_path,
        bgm_duration or target_duration
    )
    bgm_stream_url = bgm_stem.get("stream_url", "")
    drums_stream_url = stem_payload(drums_path).get("stream_url", "")
    bass_stream_url = stem_payload(bass_path).get("stream_url", "")

    prepare_instrument_audio(audio_id, detected_instruments)

    enriched_detections = []
    for item in (detected_instruments or []):
        name = item.get("instrument") or item.get("instrument_name") or "Unknown"
        conf = float(item["confidence"]) if item.get("confidence") is not None else 85.0
        
        inst_url = url_for(
            "separation.get_separated_audio",
            audio_id=audio_id,
            instrument=name
        )
        inst_dl = url_for(
            "separation.get_separated_audio",
            audio_id=audio_id,
            instrument=name,
            download="1"
        )

        enriched_detections.append({
            "instrument": name,
            "instrument_name": name,
            "confidence": round(conf, 1),
            "source": item.get("source") or item.get("source_stem") or "BGM",
            "source_stem": item.get("source_stem") or item.get("source") or "BGM",
            "bgm_stream_url": bgm_stream_url,
            "stem_url": inst_url,
            "instrument_stream_url": inst_url,
            "audio_url": inst_url,
            "download_url": inst_dl,
            "is_isolated": True,
            "solo_available": True,
            "duration": bgm_duration or target_duration
        })
 
    # -------------------------------------------------------- 
    # RESPONSE 
    # -------------------------------------------------------- 
 
    return { 
 
        "success": True, 
 
        "is_separated": True, 
 
        "message": message, 
 
        "audio_id": audio_id, 
 
        "full_song": full_song, 
 
        "vocals": stem_payload( 
            vocals_path, 
            vocals_duration 
            or target_duration 
        ), 
 
        "bgm": { 
 
            **stem_payload( 
                bgm_path, 
                bgm_duration 
                or target_duration 
            ), 
 
            "name": "BGM", 
 
            "supported_instruments": 
                supported_instruments, 
 
            "instrument_count": 
                len( 
                    supported_instruments 
                ) 
        }, 
 
        # ---------------------------------------------------- 
        # REAL AI DETECTIONS 
        # ---------------------------------------------------- 
 
        "detected_instruments": 
            enriched_detections, 
 
        "detected_instrument_count": 
            len( 
                enriched_detections 
            ), 
 
        # ---------------------------------------------------- 
        # STEM PATHS 
        # ---------------------------------------------------- 
 
        "stems": { 
 
            "vocals": 
                vocals_path, 
 
            "bgm": 
                bgm_path, 
 
            "drums": 
                drums_path, 
 
            "bass": 
                bass_path, 
 
            "other": 
                other_path 
        }, 
 
        # ---------------------------------------------------- 
        # STEM URLS 
        # ---------------------------------------------------- 
 
        "stem_urls": { 
 
            name: stem_payload(path) 
 
            for name, path in { 
 
                "vocals": 
                    vocals_path, 
 
                "bgm": 
                    bgm_path, 
 
                "drums": 
                    drums_path, 
 
                "bass": 
                    bass_path, 
 
                "other": 
                    other_path 
 
            }.items() 
        } 
    } 
 
 
# ============================================================
# CHECK / RESTORE SEPARATION
# ============================================================

def _save_separation_record(audio_id, stems):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM separation_results WHERE audio_id = %s", (audio_id,))
        cur.execute(
            """
            INSERT INTO separation_results
            (audio_id, vocals_path, drums_path, bass_path, other_path)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                audio_id,
                stems.get("vocals"),
                stems.get("drums") or None,
                stems.get("bass") or None,
                stems.get("other") or None,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as error:
        print("DB error saving separation record:", error)


def check_or_restore_separation(audio_id, file_path):
    """Return existing stems for this audio (and remember where they are)."""
    separated_folder = os.path.abspath(current_app.config["SEPARATED_FOLDER"])
    stems = find_stems_for_audio(audio_id, file_path, separated_folder)
    if not stems:
        return None

    # Record the location so instrument playback, transcription and
    # performance pages can always find these stems again.
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT vocals_path FROM separation_results WHERE audio_id = %s ORDER BY separation_id DESC LIMIT 1",
            (audio_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or row.get("vocals_path") != stems.get("vocals"):
            _save_separation_record(audio_id, stems)
    except Exception as error:
        print("Error checking DB separation:", error)

    return stems


# ============================================================ 
# DATASET AUDIO 
# ============================================================ 
 
@separation_bp.route( 
    "/datasets", 
    methods=["GET"] 
) 
def list_dataset_audio(): 
 
    dataset_folder = current_app.config[ 
        "DATASET_FOLDER" 
    ] 
 
    files = [] 
 
    for root, _, names in os.walk( 
        dataset_folder 
    ): 
 
        for name in names: 
 
            if ( 
                os.path.splitext(name)[1].lower() 
                in DATASET_EXTENSIONS 
            ): 
 
                full_path = os.path.join( 
                    root, 
                    name 
                ) 
 
                files.append({ 
 
                    "name": 
                        name, 
 
                    "path": 
                        os.path.relpath( 
                            full_path, 
                            dataset_folder 
                        ).replace( 
                            os.sep, 
                            "/" 
                        ), 
 
                    "size": 
                        os.path.getsize( 
                            full_path 
                        ) 
                }) 
 
    return jsonify({ 
 
        "success": True, 
 
        "datasets": 
            sorted( 
                files, 
                key=lambda item: 
                    item["path"].lower() 
            ), 
 
        "message": 
            ( 
                "No dataset audio files found" 
                if not files 
                else 
                "Dataset audio loaded" 
            ) 
 
    }), 200 
 
 
# ============================================================ 
# DATASET AUDIO FILE 
# ============================================================ 
 
@separation_bp.route( 
    "/datasets/file", 
    methods=["GET"] 
) 
def get_dataset_audio(): 
 
    relative_path = request.args.get( 
        "path", 
        "" 
    ) 
 
    dataset_folder = os.path.abspath( 
        current_app.config[ 
            "DATASET_FOLDER" 
        ] 
    ) 
 
    requested_path = os.path.abspath( 
        os.path.join( 
            dataset_folder, 
            relative_path 
        ) 
    ) 
 
    if not os.path.normcase( 
        requested_path 
    ).startswith( 
        os.path.normcase( 
            dataset_folder 
        ) + os.sep 
    ): 
 
        return jsonify({ 
 
            "success": False, 
 
            "message": 
                "Invalid dataset path" 
 
        }), 400 
 
    if not os.path.isfile( 
        requested_path 
    ): 
 
        return jsonify({ 
 
            "success": False, 
 
            "message": 
                "Dataset audio not found" 
 
        }), 404 
 
    return send_file( 
        requested_path, 
        as_attachment=False, 
        conditional=True 
    ) 
 
 
# ============================================================ 
# SEPARATED AUDIO STREAM 
# ============================================================ 
 
@separation_bp.route( 
    "/file", 
    methods=["GET"] 
) 
def get_separated_audio(): 
 
    file_path = request.args.get( 
        "path", 
        "" 
    ).strip() 
    audio_id = request.args.get("audio_id")
    stem = (request.args.get("stem") or "").lower().strip()
    instrument = (request.args.get("instrument") or "").strip()
 
    separated_folder = os.path.abspath( 
        current_app.config[ 
            "SEPARATED_FOLDER" 
        ] 
    ) 
 
    requested_path = None

    if audio_id and instrument:
        requested_path = get_or_create_isolated_instrument(audio_id, instrument)

    if not requested_path and audio_id and stem and stem not in ("vocals", "bgm", "accompaniment", "drums", "bass", "other"):
        requested_path = get_or_create_isolated_instrument(audio_id, stem)

    if not requested_path and audio_id:
        stems = find_stems_for_audio(audio_id, separated_root=separated_folder) or {}
        if stem in ("bgm", "accompaniment"):
            requested_path = stems.get("bgm") or stems.get("other")
        elif stem:
            requested_path = stems.get(stem)

    if not requested_path and file_path:
        requested_path = os.path.abspath(file_path)

    if not requested_path:
        if audio_id:
            return jsonify({
                "success": False,
                "message": "This song has not been separated yet. Run AI separation first."
            }), 404
        return jsonify({
            "success": False,
            "message": "No separated audio path or audio_id provided"
        }), 400

    if os.path.splitext(requested_path)[1].lower() not in DATASET_EXTENSIONS:
        return jsonify({"success": False, "message": "Only audio files can be streamed"}), 400

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root_separated = os.path.abspath(os.path.join(project_root, "separated"))
    upload_folder = os.path.abspath(current_app.config.get("UPLOAD_FOLDER", ""))

    valid_prefix = False
    for allowed_dir in [separated_folder, root_separated, upload_folder, project_root]:
        if allowed_dir and os.path.normcase(requested_path).startswith(os.path.normcase(allowed_dir) + os.sep):
            valid_prefix = True
            break

    if not valid_prefix or not os.path.isfile(requested_path):
        if not os.path.isfile(requested_path):
            return jsonify({
                "success": False,
                "message": "Separated audio not found"
            }), 404
        return jsonify({
            "success": False,
            "message": "Invalid separated audio path"
        }), 400

    lower_ext = os.path.splitext(requested_path)[1].lower()
    mimetype = "audio/wav" if lower_ext == ".wav" else ("audio/mpeg" if lower_ext == ".mp3" else None)

    response = send_file(
        requested_path,
        as_attachment=(
            request.args.get("download") == "1"
        ),
        download_name=(
            f"{secure_filename(instrument) or 'instrument'}.wav"
            if instrument else os.path.basename(requested_path)
        ),
        conditional=True,
        mimetype=mimetype
    )
    response.headers["Accept-Ranges"] = "bytes"
    return response 


# ============================================================
# DEDICATED ISOLATED INSTRUMENT AUDIO ROUTE
# ============================================================

@separation_bp.route(
    "/instrument_audio/<int:audio_id>/<instrument_name>",
    methods=["GET"]
)
def get_isolated_instrument_endpoint(audio_id, instrument_name):
    isolated_path = get_or_create_isolated_instrument(audio_id, instrument_name)
    if not isolated_path or not os.path.isfile(isolated_path):
        return jsonify({
            "success": False,
            "message": f"Could not isolate instrument {instrument_name}"
        }), 404

    response = send_file(
        isolated_path,
        as_attachment=(request.args.get("download") == "1"),
        download_name=f"{instrument_name}_{os.path.basename(isolated_path)}",
        conditional=True,
        mimetype="audio/wav"
    )
    response.headers["Accept-Ranges"] = "bytes"
    return response
 
 
# ============================================================ 
# LIST UPLOADED SONGS 
# ============================================================ 
 
@separation_bp.route( 
    "/list", 
    methods=["GET"] 
) 
def list_uploaded_songs(): 
 
    try: 
 
        connection = get_connection() 
 
        cursor = connection.cursor( 
            dictionary=True 
        ) 
 
        cursor.execute( 
            """ 
            SELECT 
                a.audio_id, 
                a.original_name, 
                a.file_path, 
                a.uploaded_at, 
                s.separation_id, 
                s.vocals_path, 
                s.drums_path, 
                s.bass_path, 
                s.other_path 
            FROM audio_files a 
            LEFT JOIN separation_results s 
                ON a.audio_id = s.audio_id 
            ORDER BY a.audio_id DESC 
            """ 
        ) 
 
        rows = cursor.fetchall() 
 
        cursor.close() 
        connection.close() 
 
        seen_ids = set() 
 
        songs = [] 
 
        separated_folder = os.path.abspath( 
            current_app.config[ 
                "SEPARATED_FOLDER" 
            ] 
        ) 
 
        for r in rows: 
 
            aid = r["audio_id"] 
 
            if aid in seen_ids: 
                continue 
 
            seen_ids.add(aid) 
 
            file_exists = bool( 
                r["file_path"] 
                and os.path.isfile( 
                    r["file_path"] 
                ) 
            ) 
 
            is_sep = bool( 
                r.get("vocals_path") 
                and os.path.isfile( 
                    r["vocals_path"] 
                ) 
            ) 
 
            if not is_sep: 
 
                disk_folder = os.path.join( 
                    separated_folder, 
                    str(aid) 
                ) 
 
                stems = scan_existing_stems( 
                    disk_folder 
                ) 
 
                if ( 
                    stems.get("vocals") 
                    and os.path.isfile( 
                        stems["vocals"] 
                    ) 
                ): 
 
                    is_sep = True 
 
            duration = ( 
                get_audio_duration( 
                    r["file_path"] 
                ) 
                if file_exists 
                else None 
            ) 
 
            songs.append({ 
 
                "audio_id": 
                    aid, 
 
                "original_name": 
                    r["original_name"], 
 
                "file_path": 
                    r["file_path"], 
 
                "duration": 
                    duration, 
 
                "is_separated": 
                    is_sep, 
 
                "uploaded_at": 
                    ( 
                        str(r["uploaded_at"]) 
                        if r.get("uploaded_at") 
                        else None 
                    ) 
            }) 
 
        return jsonify({ 
 
            "success": True, 
 
            "songs": 
                songs, 
 
            "total": 
                len(songs) 
 
        }), 200 
 
    except Exception as error: 
 
        return jsonify({ 
 
            "success": False, 
 
            "message": 
                "Failed to fetch songs list", 
 
            "error": 
                str(error) 
 
        }), 500 
 
 
# ============================================================ 
# SEPARATION STATUS 
# ============================================================ 
 
@separation_bp.route( 
    "/status/<int:audio_id>", 
    methods=["GET"] 
) 
def get_separation_status( 
    audio_id 
): 
 
    try: 
 
        connection = get_connection() 
 
        cursor = connection.cursor( 
            dictionary=True 
        ) 
 
        cursor.execute( 
            """ 
            SELECT * 
            FROM audio_files 
            WHERE audio_id = %s 
            """, 
            (audio_id,) 
        ) 
 
        audio = cursor.fetchone() 
 
        cursor.close() 
        connection.close() 
 
        if not audio: 
 
            return jsonify({ 
 
                "success": False, 
 
                "is_separated": 
                    False, 
 
                "message": 
                    "Audio not found" 
 
            }), 404 
 
        stems = check_or_restore_separation( 
            audio_id, 
            audio["file_path"] 
        ) 
 
        if ( 
            stems 
            and stems.get("vocals") 
            and ( 
                stems.get("bgm") 
                or stems.get("other") 
            ) 
        ): 
 
            bgm_path = ( 
                stems.get("bgm") 
                or stems.get("other") 
            ) 
 
            detections = get_or_detect_bgm_instruments(audio_id, bgm_path)

            payload = build_separation_payload( 
 
                audio_id, 
 
                audio["file_path"], 
 
                stems, 
 
                message= 
                    "Separation loaded successfully", 
 
                detected_instruments= 
                    detections 
            ) 
 
            return jsonify( 
                payload 
            ), 200 
 
        return jsonify({ 
 
            "success": 
                True, 
 
            "is_separated": 
                False, 
 
            "audio_id": 
                audio_id, 
 
            "original_name": 
                audio["original_name"], 
 
            "file_path": 
                audio["file_path"], 
 
            "duration": 
                get_audio_duration( 
                    audio["file_path"] 
                ), 
 
            "message": 
                "Audio not yet separated" 
 
        }), 200 
 
    except Exception as error: 
 
        print( 
            "Status Error:", 
            error 
        ) 
 
        return jsonify({ 
 
            "success": 
                False, 
 
            "message": 
                "Failed to check status", 
 
            "error": 
                str(error) 
 
        }), 500 
 
 
# ============================================================ 
# RUN SEPARATION 
# ============================================================ 
 
@separation_bp.route( 
    "/run", 
    methods=["POST"] 
) 
def run_separation(): 
 
    data = request.get_json() 
 
    if not data: 
 
        return jsonify({ 
 
            "success": 
                False, 
 
            "message": 
                "No data received" 
 
        }), 400 
 
    audio_id = data.get( 
        "audio_id" 
    ) 
 
    force = bool( 
        data.get( 
            "force", 
            False 
        ) 
    ) 
 
    if not audio_id: 
 
        return jsonify({ 
 
            "success": 
                False, 
 
            "message": 
                "audio_id is required" 
 
        }), 400 
 
    try: 
 
        # ---------------------------------------------------- 
        # GET AUDIO 
        # ---------------------------------------------------- 
 
        connection = get_connection() 
 
        cursor = connection.cursor( 
            dictionary=True 
        ) 
 
        cursor.execute( 
            """ 
            SELECT * 
            FROM audio_files 
            WHERE audio_id = %s 
            """, 
            (audio_id,) 
        ) 
 
        audio = cursor.fetchone() 
 
        cursor.close() 
        connection.close() 
 
        if not audio: 
 
            return jsonify({ 
 
                "success": 
                    False, 
 
                "message": 
                    "Audio file not found" 
 
            }), 404 
 
        # ---------------------------------------------------- 
        # CACHE CHECK 
        # ---------------------------------------------------- 
 
        if not force: 
 
            cached_stems = ( 
                check_or_restore_separation( 
                    audio_id, 
                    audio["file_path"] 
                ) 
            ) 
 
            if ( 
                cached_stems 
                and cached_stems.get("vocals") 
                and ( 
                    cached_stems.get("bgm") 
                    or cached_stems.get("other") 
                ) 
            ): 
 
                print( 
                    f"Returning cached stems " 
                    f"for audio_id {audio_id}" 
                ) 
 
                bgm_path = ( 
                    cached_stems.get("bgm") 
                    or cached_stems.get("other") 
                ) 
 
                detections = get_or_detect_bgm_instruments(audio_id, bgm_path)

                return jsonify( 
                    build_separation_payload( 
 
                        audio_id, 
 
                        audio["file_path"], 
 
                        cached_stems, 
 
                        message= 
                            "Cached separation loaded successfully", 
 
                        detected_instruments= 
                            detections 
                    ) 
                ), 200 
 
        # ---------------------------------------------------- 
        # RUN DEMUCS 
        # ---------------------------------------------------- 
 
        print() 
        print( 
            "======================================" 
        ) 
        print( 
            f"Running AI separation " 
            f"for audio_id {audio_id}" 
        ) 
        print( 
            "======================================" 
        ) 
 
        try:
            result = separate_audio(
                audio["file_path"],
                audio_id,
                force=force
            )
        except SeparationInProgressError as error:
            return jsonify({
                "success": True,
                "is_separated": False,
                "in_progress": True,
                "audio_id": audio_id,
                "message": str(error)
            }), 202
 
        vocals_path = result.get( 
            "vocals" 
        ) 
 
        drums_path = result.get( 
            "drums" 
        ) 
 
        bass_path = result.get( 
            "bass" 
        ) 
 
        other_path = result.get( 
            "other" 
        ) 
 
        bgm_path = ( 
            result.get("bgm") 
            or other_path 
        ) 
 
        # ---------------------------------------------------- 
        # SAVE SEPARATION RESULT 
        # ---------------------------------------------------- 
 
        try: 
 
            conn = get_connection() 
 
            cur = conn.cursor() 
 
            cur.execute( 
                """ 
                DELETE FROM separation_results 
                WHERE audio_id = %s 
                """, 
                (audio_id,) 
            ) 
 
            cur.execute( 
                """ 
                INSERT INTO separation_results 
                ( 
                    audio_id, 
                    vocals_path, 
                    drums_path, 
                    bass_path, 
                    other_path 
                ) 
                VALUES (%s, %s, %s, %s, %s) 
                """, 
                ( 
                    audio_id, 
                    vocals_path, 
                    drums_path, 
                    bass_path, 
                    other_path 
                ) 
            ) 
 
            conn.commit() 
 
            cur.close() 
            conn.close() 
 
        except Exception as db_error: 
 
            print( 
                "Database separation save warning:", 
                db_error 
            ) 
 
        # ---------------------------------------------------- 
        # REAL AI INSTRUMENT DETECTION 
        # ---------------------------------------------------- 
 
        detections = ( 
            detect_bgm_instruments( 
                bgm_path 
            ) 
        ) 
 
        # ---------------------------------------------------- 
        # SAVE INSTRUMENT RESULTS 
        # ---------------------------------------------------- 
 
        save_instrument_detections( 
            audio_id, 
            detections 
        ) 
 
        # ---------------------------------------------------- 
        # RESPONSE 
        # ---------------------------------------------------- 
 
        payload = build_separation_payload( 
 
            audio_id, 
 
            audio["file_path"], 
 
            result, 
 
            message= 
                "Vocals, BGM and instruments analyzed successfully", 
 
            detected_instruments= 
                detections 
        ) 
 
        return jsonify( 
            payload 
        ), 200 
 
    except Exception as error: 
 
        print( 
            "Separation Error:", 
            error 
        ) 
 
        return jsonify({ 
 
            "success": 
                False, 
 
            "message": 
                "Audio separation failed", 
 
            "error": 
                str(error) 
 
        }), 500 
 
 
# ============================================================ 
# GET DETECTED INSTRUMENTS 
# ============================================================ 
 
@separation_bp.route( 
    "/instruments/<int:audio_id>", 
    methods=["GET"] 
) 
def get_detected_instruments( 
    audio_id 
): 
 
    try: 
 
        connection = get_connection() 
 
        cursor = connection.cursor( 
            dictionary=True 
        ) 
 
        cursor.execute( 
            """ 
            SELECT 
                detection_id, 
                audio_id, 
                instrument_name, 
                confidence, 
                source_stem, 
                created_at 
            FROM instrument_detections 
            WHERE audio_id = %s 
            ORDER BY detection_id ASC 
            """, 
            (audio_id,) 
        ) 
 
        rows = cursor.fetchall() 
 
        cursor.close() 
        connection.close() 

        if not rows:
            audio_connection = get_connection()
            audio_cursor = audio_connection.cursor(dictionary=True)
            audio_cursor.execute(
                "SELECT file_path FROM audio_files WHERE audio_id = %s",
                (audio_id,)
            )
            audio = audio_cursor.fetchone()
            audio_cursor.close()
            audio_connection.close()

            if audio:
                stems = check_or_restore_separation(audio_id, audio.get("file_path"))
                bgm_path = (stems or {}).get("bgm") or (stems or {}).get("other")
                if bgm_path:
                    detections = detect_bgm_instruments(bgm_path)
                    if detections:
                        save_instrument_detections(audio_id, detections)
                        rows = [
                            {
                                "detection_id": None,
                                "audio_id": audio_id,
                                "instrument_name": item.get("instrument"),
                                "confidence": item.get("confidence"),
                                "source_stem": item.get("source", "BGM"),
                                "created_at": None,
                            }
                            for item in detections
                        ]

        clean_rows = []
        for r in rows:
            inst_name = r["instrument_name"]
            inst_url = url_for(
                "separation.get_separated_audio",
                audio_id=audio_id,
                instrument=inst_name
            )
            clean_rows.append({
                "detection_id": r["detection_id"],
                "audio_id": r["audio_id"],
                "instrument_name": inst_name,
                "confidence": float(r["confidence"]) if r.get("confidence") is not None else None,
                "source_stem": r.get("source_stem"),
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
                "audio_url": inst_url,
                "instrument_stream_url": inst_url
            })
 
        return jsonify({ 
 
            "success": 
                True, 
 
            "audio_id": 
                audio_id, 
 
            "instruments": 
                clean_rows, 
 
            "count": 
                len(clean_rows) 
 
        }), 200 
 
    except Exception as error: 
 
        return jsonify({ 
 
            "success": 
                False, 
 
            "message": 
                "Could not fetch detected instruments", 
 
            "error": 
                str(error) 
 
        }), 500     