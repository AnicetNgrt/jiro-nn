use crate::ops::{
    op_graphs::op_node::OpNodeTrait, op_graphs_builders::op_node_builder::OpNodeBuilder, Data,
};

use super::{
    input_mapping::InputMappingOp, reference_mapping::ReferenceMappingOp,
    total_mapping::TotalMappingOp,
};

pub trait MappingBuilderMode {}

pub struct InputMappingBuilder;
impl InputMappingBuilder {
    pub fn new<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>(
        mapper: F,
        mapper_reversed: FP,
        meta_mapper: FM,
    ) -> MappingBuilder<'g, DataIn, DataOut, F, FP, FM, Self>
    where
        F: Fn(DataIn) -> DataOut + 'g,
        FP: Fn(DataOut) -> DataIn + 'g,
        FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
    {
        MappingBuilder::new(mapper, mapper_reversed, meta_mapper)
    }
}
impl MappingBuilderMode for InputMappingBuilder {}

pub struct ReferenceMappingBuilder;
impl ReferenceMappingBuilder {
    pub fn new<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>(
        mapper: F,
        mapper_reversed: FP,
        meta_mapper: FM,
    ) -> MappingBuilder<'g, DataIn, DataOut, F, FP, FM, Self>
    where
        F: Fn(DataIn) -> DataOut + 'g,
        FP: Fn(DataOut) -> DataIn + 'g,
        FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
    {
        MappingBuilder::new(mapper, mapper_reversed, meta_mapper)
    }
}
impl MappingBuilderMode for ReferenceMappingBuilder {}

pub struct TotalMappingBuilder;
impl TotalMappingBuilder {
    pub fn new<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>(
        mapper: F,
        mapper_reversed: FP,
        meta_mapper: FM,
    ) -> MappingBuilder<'g, DataIn, DataOut, F, FP, FM, Self>
    where
        F: Fn(DataIn) -> DataOut + 'g,
        FP: Fn(DataOut) -> DataIn + 'g,
        FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
    {
        MappingBuilder::new(mapper, mapper_reversed, meta_mapper)
    }
}
impl MappingBuilderMode for TotalMappingBuilder {}

pub struct MappingBuilder<
    'g,
    DataIn: Data<'g>,
    DataOut: Data<'g>,
    F,
    FP,
    FM,
    Mode: MappingBuilderMode,
> where
    F: Fn(DataIn) -> DataOut + 'g,
    FP: Fn(DataOut) -> DataIn + 'g,
    FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
{
    f: Option<F>,
    fp: Option<FP>,
    fm: Option<FM>,
    _phantom: std::marker::PhantomData<&'g (DataIn, DataOut, Mode)>,
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM, Mode: MappingBuilderMode>
    MappingBuilder<'g, DataIn, DataOut, F, FP, FM, Mode>
where
    F: Fn(DataIn) -> DataOut + 'g,
    FP: Fn(DataOut) -> DataIn + 'g,
    FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
{
    pub fn new(mapper: F, mapper_reversed: FP, meta_mapper: FM) -> Self {
        Self {
            f: Some(mapper),
            fp: Some(mapper_reversed),
            fm: Some(meta_mapper),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, DataRef: Data<'g>, F, FP, FM>
    OpNodeBuilder<'g, DataIn, DataOut, DataRef, DataRef>
    for MappingBuilder<'g, DataIn, DataOut, F, FP, FM, InputMappingBuilder>
where
    F: Fn(DataIn) -> DataOut + 'g,
    FP: Fn(DataOut) -> DataIn + 'g,
    FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataRef::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataRef, DataRef> + 'g>,
        (DataOut::Meta, DataRef::Meta),
    ) {
        let meta_out = self.fm.as_ref().expect("Mapping builder built twice.")(meta_data);
        let op = InputMappingOp::new(
            self.f.take().expect("Mapping builder built twice."),
            self.fp.take().expect("Mapping builder built twice."),
            self.fm.take().expect("Mapping builder built twice."),
        );
        (Box::new(op), (meta_out, meta_ref))
    }
}

impl<'g, DataRefIn: Data<'g>, DataRefOut: Data<'g>, D: Data<'g>, F, FP, FM>
    OpNodeBuilder<'g, D, D, DataRefIn, DataRefOut>
    for MappingBuilder<'g, DataRefIn, DataRefOut, F, FP, FM, ReferenceMappingBuilder>
where
    F: Fn(DataRefIn) -> DataRefOut + 'g,
    FP: Fn(DataRefOut) -> DataRefIn + 'g,
    FM: Fn(DataRefIn::Meta) -> DataRefOut::Meta + 'g,
{
    fn build(
        &mut self,
        meta_data: D::Meta,
        meta_ref: DataRefIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, D, D, DataRefIn, DataRefOut> + 'g>,
        (D::Meta, DataRefOut::Meta),
    ) {
        let meta_out = self.fm.as_ref().expect("Mapping builder built twice.")(meta_ref);
        let op = ReferenceMappingOp::new(
            self.f.take().expect("Mapping builder built twice."),
            self.fp.take().expect("Mapping builder built twice."),
            self.fm.take().expect("Mapping builder built twice."),
        );
        (Box::new(op), (meta_data, meta_out))
    }
}

impl<'g, DataIn: Data<'g>, DataOut: Data<'g>, F, FP, FM>
    OpNodeBuilder<'g, DataIn, DataOut, DataIn, DataOut>
    for MappingBuilder<'g, DataIn, DataOut, F, FP, FM, TotalMappingBuilder>
where
    F: Fn(DataIn) -> DataOut + 'g,
    FP: Fn(DataOut) -> DataIn + 'g,
    FM: Fn(DataIn::Meta) -> DataOut::Meta + 'g,
{
    fn build(
        &mut self,
        meta_data: DataIn::Meta,
        meta_ref: DataIn::Meta,
    ) -> (
        Box<dyn OpNodeTrait<'g, DataIn, DataOut, DataIn, DataOut> + 'g>,
        (DataOut::Meta, DataOut::Meta),
    ) {
        let fm = self.fm.as_ref().expect("Mapping builder built twice.");
        let meta_data_out = (fm)(meta_data);
        let meta_ref_out = (fm)(meta_ref);
        let op = TotalMappingOp::new(
            self.f.take().expect("Mapping builder built twice."),
            self.fp.take().expect("Mapping builder built twice."),
            self.fm.take().expect("Mapping builder built twice."),
        );
        (Box::new(op), (meta_data_out, meta_ref_out))
    }
}
