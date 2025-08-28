// rust/lancedb/src/c_api.rs
// Version: v1.21
// Date: 2025-08-26
// Author: Grok (assisted modeling)
// Description: Builds on v1.20 with added eprintln! logging for Err in lance_create_table to debug schema deserialization.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float};
use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;  // For as_primitive_opt
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_ipc::reader::StreamReader;  // For IPC stream reader
use arrow::record_batch::RecordBatchIterator;  // For empty batch iterator
use arrow_schema::Schema;
use futures::executor::block_on;  // For synchronous wrapping of async methods
use futures::stream::TryStreamExt;  // For try_collect on streams
use lance::dataset::ROW_ID;

use crate::connection::Connection;
use crate::error::Error as LanceDBError;
use crate::index::vector::IvfHnswPqIndexBuilder;
use crate::index::Index;
use crate::query::{QueryBase, ExecutableQuery, Select};
use crate::table::Table;

// Additional imports for JSON serialization and deletion
use arrow_json::writer::{LineDelimited, Writer};
use std::io::Cursor;

// MARK: - Constants
const DIST_COL: &str = "_distance";

// MARK: - Error Codes
/// C-compatible error code enum (0 = success, negative = error).
#[repr(C)]
pub enum LanceErrorCode {
    Success = 0,
    Failure = -1,
}

// MARK: - KNN Result Struct (replaces tuple for cbindgen compatibility)
#[repr(C)]
pub struct LanceKnnResult {
    id: u64,
    distance: c_float,
}

// MARK: - Opaque Pointer Types (for Swift)
type LanceConnectionPtr = *mut Connection;
type LanceTablePtr = *mut Table;

// MARK: - Helper Functions
/// Converts C string to Rust String, returns Err if invalid.
fn cstr_to_string(ptr: *const c_char) -> Result<String, LanceDBError> {
    unsafe {
        if ptr.is_null() {
            return Err(LanceDBError::InvalidInput { message: "Null pointer for string".to_string() });
        }
        CStr::from_ptr(ptr).to_str().map(|s| s.to_string()).map_err(|_| LanceDBError::InvalidInput { message: "Invalid UTF-8".to_string() })
    }
}

// MARK: - Connection Functions
/// Connects to LanceDB at the given path (blocks on async).
/// Returns opaque pointer to connection or null on error.
#[no_mangle]
pub extern "C" fn lance_connect(path: *const c_char) -> LanceConnectionPtr {
    let path_str = match cstr_to_string(path) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match block_on(crate::connect(&path_str).execute()) {
        Ok(conn) => Box::into_raw(Box::new(conn)),
        Err(_) => ptr::null_mut(),
    }
}

/// Frees a connection pointer.
#[no_mangle]
pub extern "C" fn lance_free_connection(conn: LanceConnectionPtr) {
    if !conn.is_null() {
        unsafe { let _ = Box::from_raw(conn); }
    }
}

// MARK: - Table Operations
/// Creates a table in the database (blocks on async).
/// Schema_json is a JSON string for the schema.
/// Returns error code.
#[no_mangle]
pub extern "C" fn lance_create_table(conn: LanceConnectionPtr, name: *const c_char, schema_json: *const c_char) -> LanceErrorCode {
    let conn = unsafe { if conn.is_null() { return LanceErrorCode::Failure; } &*conn };
    let name_str = match cstr_to_string(name) {
        Ok(s) => s,
        Err(_) => return LanceErrorCode::Failure,
    };
    let schema_str = match cstr_to_string(schema_json) {
        Ok(s) => s,
        Err(_) => return LanceErrorCode::Failure,
    };
    let schema: Schema = match serde_json::from_str(&schema_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Schema deserialization error: {:?}", e);  // Added logging for debug
            return LanceErrorCode::Failure;
        }
    };
    let schema_ref = Arc::new(schema);
    let data = RecordBatchIterator::new(vec![], schema_ref.clone()); // Empty data for creation
    match block_on(conn.create_table(&name_str, data).execute()) {
        Ok(_) => LanceErrorCode::Success,
        Err(e) => {
            eprintln!("Create table error: {:?}", e);  // Added logging for debug
            LanceErrorCode::Failure
        }
    }
}

/// Opens a table from the connection (blocks on async).
/// Returns opaque pointer to table or null on error.
#[no_mangle]
pub extern "C" fn lance_open_table(conn: LanceConnectionPtr, name: *const c_char) -> LanceTablePtr {
    let conn = unsafe { if conn.is_null() { return ptr::null_mut(); } &*conn };
    let name_str = match cstr_to_string(name) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match block_on(conn.open_table(&name_str).execute()) {
        Ok(table) => Box::into_raw(Box::new(table)),
        Err(_) => ptr::null_mut(),
    }
}

/// Frees a table pointer.
#[no_mangle]
pub extern "C" fn lance_free_table(table: LanceTablePtr) {
    if !table.is_null() {
        unsafe { let _ = Box::from_raw(table); }
    }
}

// MARK: - Add Operations
/// Adds a record batch to the table (blocks on async).
/// Batch_ptr is a pointer to serialized Arrow IPC data; length is size in bytes.
/// Returns error code.
#[no_mangle]
pub extern "C" fn lance_add_batch(table: LanceTablePtr, batch_ptr: *const u8, length: usize) -> LanceErrorCode {
    let table = unsafe { if table.is_null() { return LanceErrorCode::Failure; } &*table };
    let ipc_slice = unsafe { std::slice::from_raw_parts(batch_ptr, length) };
    let reader: Box<dyn RecordBatchReader + Send> = match StreamReader::try_new(ipc_slice, None) {
        Ok(r) => Box::new(r),
        Err(_) => return LanceErrorCode::Failure,
    };
    match block_on(table.add(reader).execute()) {
        Ok(_) => LanceErrorCode::Success,
        Err(_) => LanceErrorCode::Failure,
    }
}

// MARK: - Query Operations
/// Queries nearest neighbors for a vector (blocks on async).
/// Vector is f32 array, dim is dimension, k is limit.
/// Returns pointer to result array of LanceKnnResult or null; out_count is number of results.
/// Caller must free with lance_free_results.
#[no_mangle]
pub extern "C" fn lance_query_knn(table: LanceTablePtr, vector: *const c_float, dim: c_int, k: c_int, out_count: *mut c_int) -> *mut LanceKnnResult {
    let table = unsafe { if table.is_null() { return ptr::null_mut(); } &*table };
    let vec_slice = unsafe { std::slice::from_raw_parts(vector, dim as usize) };
    let query_builder = table.query();
    let vector_query = match query_builder.nearest_to(vec_slice) {
        Ok(q) => q,
        Err(_) => return ptr::null_mut(),
    };
    let query = vector_query.with_row_id().limit(k as usize).select(Select::All);
    let stream = match block_on(query.execute()) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let results = match block_on(stream.try_collect::<Vec<_>>()) {
        Ok(batches) => batches,
        Err(_) => return ptr::null_mut(),
    };
    let mut result_vec: Vec<LanceKnnResult> = Vec::new();
    for batch in results {
        let ids = batch.column_by_name(ROW_ID).unwrap().as_primitive_opt::<UInt64Type>().unwrap();
        let dists = batch.column_by_name(DIST_COL).unwrap().as_primitive_opt::<Float32Type>().unwrap();
        for i in 0..batch.num_rows() {
            result_vec.push(LanceKnnResult {
                id: ids.value(i),
                distance: dists.value(i),
            });
        }
    }
    unsafe { *out_count = result_vec.len() as c_int; }
    let boxed = Box::new(result_vec);
    Box::into_raw(boxed).cast::<LanceKnnResult>()
}

/// Frees a KNN results pointer.
#[no_mangle]
pub extern "C" fn lance_free_results(results: *mut LanceKnnResult) {
    if !results.is_null() {
        unsafe { let _ = Box::from_raw(results as *mut Vec<LanceKnnResult>); }
    }
}

/// Fetches a row by row_id as JSON string (blocks on async).
/// Returns allocated C string (caller must free with lance_free_string) or null on error.
#[no_mangle]
pub extern "C" fn lance_get_row_by_id(table: LanceTablePtr, id: u64) -> *const c_char {
    let table = unsafe { if table.is_null() { return ptr::null(); } &*table };
    let query = table.take_row_ids(vec![id]).with_row_id().select(Select::All);
    let stream = match block_on(query.execute()) {
        Ok(s) => s,
        Err(_) => return ptr::null(),
    };
    let results = match block_on(stream.try_collect::<Vec<_>>()) {
        Ok(batches) => batches,
        Err(_) => return ptr::null(),
    };
    if results.is_empty() || results[0].num_rows() == 0 {
        return ptr::null();
    }
    let batch = &results[0];
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = Writer::<_, LineDelimited>::new(&mut cursor);
    if writer.write(batch).is_err() {
        return ptr::null();
    }
    let json_bytes = cursor.into_inner();
    match CString::new(json_bytes) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null(),
    }
}

/// Frees a string returned by LanceDB (e.g., from lance_get_row_by_id).
#[no_mangle]
pub extern "C" fn lance_free_string(s: *const c_char) {
    if !s.is_null() {
        unsafe { let _ = CString::from_raw(s as *mut c_char); }
    }
}

// MARK: - Index Operations
/// Creates an IVF_HNSW_PQ index on the table (blocks on async).
/// Columns is C string array, num_cols is count.
/// Returns error code.
#[no_mangle]
pub extern "C" fn lance_create_hnsw_index(table: LanceTablePtr, columns: *const *const c_char, num_cols: c_int) -> LanceErrorCode {
    let table = unsafe { if table.is_null() { return LanceErrorCode::Failure; } &*table };
    let cols_slice = unsafe { std::slice::from_raw_parts(columns, num_cols as usize) };
    let mut cols_vec: Vec<String> = Vec::new();
    for &ptr in cols_slice {
        match cstr_to_string(ptr) {
            Ok(s) => cols_vec.push(s),
            Err(_) => return LanceErrorCode::Failure,
        }
    }
    let builder = IvfHnswPqIndexBuilder::default();
    match block_on(table.create_index(&cols_vec, Index::IvfHnswPq(builder)).execute()) {
        Ok(_) => LanceErrorCode::Success,
        Err(_) => LanceErrorCode::Failure,
    }
}

// MARK: - Delete Operations
/// Clears all data from the table (blocks on async, deletes where true).
/// Returns error code.
#[no_mangle]
pub extern "C" fn lance_clear_table(table: LanceTablePtr) -> LanceErrorCode {
    let table = unsafe { if table.is_null() { return LanceErrorCode::Failure; } &*table };
    match block_on(table.delete("true")) {
        Ok(_) => LanceErrorCode::Success,
        Err(_) => LanceErrorCode::Failure,
    }
}