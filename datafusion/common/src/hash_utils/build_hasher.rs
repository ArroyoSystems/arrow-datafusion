// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use super::{AsDynArray, HASH_BUFFER, MAX_BUFFER_SIZE};
#[cfg(not(feature = "force_hash_collisions"))]
use super::{
    ChildHashing, combine_hashes, hash_dictionary_with_child_hashing,
    hash_fixed_list_array, hash_list_array, hash_list_view_array, hash_map_array,
    hash_run_array, hash_struct_array, hash_union_array,
};
#[cfg(not(feature = "force_hash_collisions"))]
use crate::cast::{
    as_binary_view_array, as_boolean_array, as_fixed_size_list_array,
    as_generic_binary_array, as_large_list_array, as_large_list_view_array,
    as_list_array, as_list_view_array, as_map_array, as_string_array,
    as_string_view_array, as_struct_array, as_union_array,
};
use crate::error::Result;
use crate::error::{_internal_datafusion_err, _internal_err};
#[cfg(feature = "force_hash_collisions")]
use arrow::array::Array;
#[cfg(not(feature = "force_hash_collisions"))]
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
#[cfg(not(feature = "force_hash_collisions"))]
use arrow::array::*;
#[cfg(not(feature = "force_hash_collisions"))]
use arrow::datatypes::*;
#[cfg(not(feature = "force_hash_collisions"))]
use arrow::{downcast_dictionary_array, downcast_primitive_array};
use std::hash::BuildHasher;

pub(super) fn with_hashes_with_hasher<I, T, F, R, S>(
    arrays: I,
    hash_builder: &S,
    callback: F,
) -> Result<R>
where
    I: IntoIterator<Item = T>,
    T: AsDynArray,
    F: FnOnce(&[u64]) -> Result<R>,
    S: BuildHasher,
{
    let mut iter = arrays.into_iter().peekable();

    let required_size = match iter.peek() {
        Some(arr) => arr.as_dyn_array().len(),
        None => {
            return _internal_err!("with_hashes_with_hasher requires at least one array");
        }
    };

    HASH_BUFFER.try_with(|cell| {
        let mut buffer = cell.try_borrow_mut().map_err(|_| {
            _internal_datafusion_err!(
                "with_hashes_with_hasher cannot be called reentrantly on the same thread"
            )
        })?;

        buffer.clear();
        buffer.resize(required_size, 0);

        create_hashes_with_hasher_impl(iter, hash_builder, &mut buffer[..required_size])?;

        let result = callback(&buffer[..required_size])?;

        if buffer.capacity() > MAX_BUFFER_SIZE {
            buffer.truncate(MAX_BUFFER_SIZE);
            buffer.shrink_to_fit();
        }

        Ok(result)
    }).map_err(|_| {
        _internal_datafusion_err!(
            "with_hashes_with_hasher cannot access thread-local storage during or after thread destruction"
        )
    })?
}

pub(super) fn create_hashes_with_hasher<'a, I, T, S>(
    arrays: I,
    hash_builder: &S,
    hashes_buffer: &'a mut [u64],
) -> Result<&'a mut [u64]>
where
    I: IntoIterator<Item = T>,
    T: AsDynArray,
    S: BuildHasher,
{
    create_hashes_with_hasher_impl(arrays, hash_builder, hashes_buffer)
}

fn create_hashes_with_hasher_impl<'a, I, T, S>(
    arrays: I,
    hash_builder: &S,
    hashes_buffer: &'a mut [u64],
) -> Result<&'a mut [u64]>
where
    I: IntoIterator<Item = T>,
    T: AsDynArray,
    S: BuildHasher,
{
    for (i, array) in arrays.into_iter().enumerate() {
        let rehash = i >= 1;
        hash_single_array_with_hasher(
            array.as_dyn_array(),
            hash_builder,
            hashes_buffer,
            rehash,
        )?;
    }
    Ok(hashes_buffer)
}

#[cfg(not(feature = "force_hash_collisions"))]
struct BuildHasherChildHashing<'a, S> {
    hash_builder: &'a S,
}

#[cfg(not(feature = "force_hash_collisions"))]
impl<S: BuildHasher> ChildHashing for BuildHasherChildHashing<'_, S> {
    fn create_hashes<I, T>(&self, arrays: I, hashes_buffer: &mut [u64]) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: AsDynArray,
    {
        create_hashes_with_hasher_impl(arrays, self.hash_builder, hashes_buffer)
            .map(|_| ())
    }
}

#[cfg(not(feature = "force_hash_collisions"))]
trait BuildHasherHashValue {
    fn hash_one_with_hasher<S: BuildHasher>(&self, state: &S) -> u64;
}

#[cfg(not(feature = "force_hash_collisions"))]
impl<T: BuildHasherHashValue + ?Sized> BuildHasherHashValue for &T {
    fn hash_one_with_hasher<S: BuildHasher>(&self, state: &S) -> u64 {
        T::hash_one_with_hasher(self, state)
    }
}

macro_rules! build_hasher_hash_value {
    ($($t:ty),+) => {
        $(#[cfg(not(feature = "force_hash_collisions"))]
        impl BuildHasherHashValue for $t {
            fn hash_one_with_hasher<S: BuildHasher>(&self, state: &S) -> u64 {
                state.hash_one(self)
            }
        })+
    };
}
build_hasher_hash_value!(i8, i16, i32, i64, i128, i256, u8, u16, u32, u64, u128);
build_hasher_hash_value!(bool, str, [u8], IntervalDayTime, IntervalMonthDayNano);

macro_rules! build_hasher_hash_float_value {
    ($(($t:ty, $i:ty)),+) => {
        $(#[cfg(not(feature = "force_hash_collisions"))]
        impl BuildHasherHashValue for $t {
            fn hash_one_with_hasher<S: BuildHasher>(&self, state: &S) -> u64 {
                state.hash_one(<$i>::from_ne_bytes(self.to_ne_bytes()))
            }
        })+
    };
}
build_hasher_hash_float_value!((half::f16, u16), (f32, u32), (f64, u64));

#[cfg(not(feature = "force_hash_collisions"))]
fn hash_null_with_hasher<S: BuildHasher>(
    hash_builder: &S,
    hashes_buffer: &mut [u64],
    multi_col: bool,
) {
    if hashes_buffer.is_empty() {
        return;
    }

    let null_hash = hash_builder.hash_one(1);
    if multi_col {
        hashes_buffer.iter_mut().for_each(|hash| {
            *hash = combine_hashes(null_hash, *hash);
        })
    } else {
        hashes_buffer.fill(null_hash);
    }
}

#[cfg(not(feature = "force_hash_collisions"))]
fn hash_array_primitive_with_hasher<T, S>(
    array: &PrimitiveArray<T>,
    hash_builder: &S,
    hashes_buffer: &mut [u64],
    rehash: bool,
) where
    T: ArrowPrimitiveType<Native: BuildHasherHashValue>,
    S: BuildHasher,
{
    assert_eq!(
        hashes_buffer.len(),
        array.len(),
        "hashes_buffer and array should be of equal length"
    );

    if array.null_count() == 0 {
        if rehash {
            for (hash, &value) in hashes_buffer.iter_mut().zip(array.values().iter()) {
                *hash = combine_hashes(value.hash_one_with_hasher(hash_builder), *hash);
            }
        } else {
            for (hash, &value) in hashes_buffer.iter_mut().zip(array.values().iter()) {
                *hash = value.hash_one_with_hasher(hash_builder);
            }
        }
    } else if rehash {
        for i in array.nulls().unwrap().valid_indices() {
            let value = unsafe { array.value_unchecked(i) };
            hashes_buffer[i] = combine_hashes(
                value.hash_one_with_hasher(hash_builder),
                hashes_buffer[i],
            );
        }
    } else {
        for i in array.nulls().unwrap().valid_indices() {
            let value = unsafe { array.value_unchecked(i) };
            hashes_buffer[i] = value.hash_one_with_hasher(hash_builder);
        }
    }
}

#[cfg(not(feature = "force_hash_collisions"))]
fn hash_array_with_hasher<T, S>(
    array: &T,
    hash_builder: &S,
    hashes_buffer: &mut [u64],
    rehash: bool,
) where
    T: ArrayAccessor,
    T::Item: BuildHasherHashValue,
    S: BuildHasher,
{
    assert_eq!(
        hashes_buffer.len(),
        array.len(),
        "hashes_buffer and array should be of equal length"
    );

    if array.null_count() == 0 {
        if rehash {
            for (i, hash) in hashes_buffer.iter_mut().enumerate() {
                let value = unsafe { array.value_unchecked(i) };
                *hash = combine_hashes(value.hash_one_with_hasher(hash_builder), *hash);
            }
        } else {
            for (i, hash) in hashes_buffer.iter_mut().enumerate() {
                let value = unsafe { array.value_unchecked(i) };
                *hash = value.hash_one_with_hasher(hash_builder);
            }
        }
    } else if rehash {
        for i in array.nulls().unwrap().valid_indices() {
            let value = unsafe { array.value_unchecked(i) };
            hashes_buffer[i] = combine_hashes(
                value.hash_one_with_hasher(hash_builder),
                hashes_buffer[i],
            );
        }
    } else {
        for i in array.nulls().unwrap().valid_indices() {
            let value = unsafe { array.value_unchecked(i) };
            hashes_buffer[i] = value.hash_one_with_hasher(hash_builder);
        }
    }
}

#[cfg(not(feature = "force_hash_collisions"))]
fn hash_single_array_with_hasher<S: BuildHasher>(
    array: &dyn Array,
    hash_builder: &S,
    hashes_buffer: &mut [u64],
    rehash: bool,
) -> Result<()> {
    let child_hashing = BuildHasherChildHashing { hash_builder };

    downcast_primitive_array! {
        array => hash_array_primitive_with_hasher(array, hash_builder, hashes_buffer, rehash),
        DataType::Null => hash_null_with_hasher(hash_builder, hashes_buffer, rehash),
        DataType::Boolean => hash_array_with_hasher(&as_boolean_array(array)?, hash_builder, hashes_buffer, rehash),
        DataType::Utf8 => hash_array_with_hasher(&as_string_array(array)?, hash_builder, hashes_buffer, rehash),
        DataType::Utf8View => hash_array_with_hasher(&as_string_view_array(array)?, hash_builder, hashes_buffer, rehash),
        DataType::LargeUtf8 => hash_array_with_hasher(&as_largestring_array(array), hash_builder, hashes_buffer, rehash),
        DataType::Binary => hash_array_with_hasher(&as_generic_binary_array::<i32>(array)?, hash_builder, hashes_buffer, rehash),
        DataType::BinaryView => hash_array_with_hasher(&as_binary_view_array(array)?, hash_builder, hashes_buffer, rehash),
        DataType::LargeBinary => hash_array_with_hasher(&as_generic_binary_array::<i64>(array)?, hash_builder, hashes_buffer, rehash),
        DataType::FixedSizeBinary(_) => {
            let array: &FixedSizeBinaryArray = array.as_any().downcast_ref().unwrap();
            hash_array_with_hasher(&array, hash_builder, hashes_buffer, rehash)
        }
        DataType::Dictionary(_, _) => downcast_dictionary_array! {
            array => hash_dictionary_with_child_hashing(array, &child_hashing, hashes_buffer, rehash)?,
            _ => unreachable!()
        }
        DataType::Struct(_) => {
            let array = as_struct_array(array)?;
            hash_struct_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::List(_) => {
            let array = as_list_array(array)?;
            hash_list_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::LargeList(_) => {
            let array = as_large_list_array(array)?;
            hash_list_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::ListView(_) => {
            let array = as_list_view_array(array)?;
            hash_list_view_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::LargeListView(_) => {
            let array = as_large_list_view_array(array)?;
            hash_list_view_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::Map(_, _) => {
            let array = as_map_array(array)?;
            hash_map_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::FixedSizeList(_,_) => {
            let array = as_fixed_size_list_array(array)?;
            hash_fixed_list_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::Union(_, _) => {
            let array = as_union_array(array)?;
            hash_union_array(array, &child_hashing, hashes_buffer)?;
        }
        DataType::RunEndEncoded(_, _) => downcast_run_array! {
            array => hash_run_array(array, &child_hashing, hashes_buffer, rehash)?,
            _ => unreachable!()
        }
        _ => {
            return _internal_err!(
                "Unsupported data type in hasher: {}",
                array.data_type()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "force_hash_collisions")]
fn hash_single_array_with_hasher<S: BuildHasher>(
    _array: &dyn Array,
    _hash_builder: &S,
    hashes_buffer: &mut [u64],
    _rehash: bool,
) -> Result<()> {
    for hash in hashes_buffer.iter_mut() {
        *hash = 0;
    }
    Ok(())
}
