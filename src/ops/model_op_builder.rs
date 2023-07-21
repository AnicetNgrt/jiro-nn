use super::{combinatory_op::{OriginOp, OpGraph}, Data, ModelOp, OpChain, mapping::{InputMappingOp, ReferenceMappingOp}};

pub struct OpOriginBuilder<D: Data, DataRef: Data> {
    _phantom: std::marker::PhantomData<(D, DataRef)>,
}

impl<D: Data, DataRef: Data> OpOriginBuilder<D, DataRef> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Data, DataRef: Data> OpBuilder<D, D, DataRef, DataRef> for OpOriginBuilder<D, DataRef> {
    fn build(
        &mut self,
        sample_data: D,
        sample_ref: DataRef,
    ) -> (
        Box<dyn ModelOp<D, D, DataRef, DataRef>>,
        (D, DataRef),
    ) {
        (
            Box::new(OriginOp::new()),
            (sample_data, sample_ref),
        )
    }
}

pub struct OpBuild<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    pub builder: Option<Box<
        dyn FnOnce(
                DataIn,
                DataRefIn,
            ) -> (
                Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
                (DataIn, DataRefIn),
            ) + 'a,
    >>,
}

impl<'a, D: Data + Clone, DataRef: Data + Clone> OpBuild<'a, (), D, (), DataRef> {
    pub fn from_data(data: D, reference: DataRef) -> Self {
        let origin_op = OriginOp::<(), ()>::new();
        let chain_op = OpChain::new(
            Box::new(origin_op),
            Box::new(InputMappingOp::new(
                move |_| data.clone(),
                |_| () 
            ))
        );
        let chain_op = OpChain::new(
            Box::new(chain_op),
            Box::new(ReferenceMappingOp::new(
                move |_| reference.clone(),
                |_| () 
            ))
        );

        Self {
            builder: Some(Box::new(move |_, _| {
                (
                    Box::new(chain_op),
                    ((), ()),
                )
            })),
        }
    }

    pub fn build_full_graph(
        &mut self
    ) -> OpGraph<'a, D, DataRef> {
        match self.builder.take() {
            None => panic!("OpBuild::build() called twice."),
            Some(builder) => OpGraph((builder)((), ()).0),
        }
    }
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    pub fn new_fn(
        builder: Box<dyn FnOnce(
            DataIn,
            DataRefIn,
        ) -> (
            Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
            (DataIn, DataRefIn),
        ) + 'a>,
    ) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    pub fn new_op_builder<OpB: OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut> + 'a>(
        mut builder: OpB,
    ) -> Self {
        Self {
            builder: Some(Box::new(move |sample_data, sample_ref| {
                builder.build(sample_data, sample_ref)
            })),
        }
    }
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data>
    OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuild<'a, DataIn, DataOut, DataRefIn, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    ) {
        match self.builder.take() {
            None => panic!("OpBuild::build() called twice."),
            Some(builder) => (builder)(sample_data, sample_ref),
        }
    }
}

macro_rules! plug_builder_on_opbuild_data_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'a, DataIn: Data, DataRefIn: Data, DataRefOut: Data>
            OpBuild<'a, DataIn, $plug_type, DataRefIn, DataRefOut>
        {
            pub fn $plug_name(self) -> OpBuild<'a, DataIn, $out_type, DataRefIn, DataRefOut> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_data_out;

macro_rules! plug_builder_on_opbuild_reference_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data>
            OpBuild<'a, DataIn, DataOut, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpBuild<'a, DataIn, DataOut, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_reference_out;

macro_rules! plug_builder_on_opbuild_total_out {
    ($plug_name:ident, $plug_type:ty, $out_type:ty, $builder:expr) => {
        impl<'a, DataIn: Data, DataRefIn: Data>
            OpBuild<'a, DataIn, $plug_type, DataRefIn, $plug_type>
        {
            pub fn $plug_name(self) -> OpBuild<'a, DataIn, $out_type, DataRefIn, $out_type> {
                self.push_and_pack($builder)
            }
        }
    };
}

pub(crate) use plug_builder_on_opbuild_total_out;

pub trait OpBuilder<DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    );
}

pub struct OpBuilderChain<
    'a,
    DataIn: Data,
    DataMid: Data,
    DataOut: Data,
    DataRefIn: Data,
    DataRefMid: Data,
    DataRefOut: Data,
> {
    first_op: Box<dyn OpBuilder<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
    second_op: Box<dyn OpBuilder<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > OpBuilderChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    pub fn new(
        first_op: Box<dyn OpBuilder<DataIn, DataMid, DataRefIn, DataRefMid> + 'a>,
        second_op: Box<dyn OpBuilder<DataMid, DataOut, DataRefMid, DataRefOut> + 'a>,
    ) -> Self {
        Self {
            first_op,
            second_op,
        }
    }
}

impl<
        'a,
        DataIn: Data,
        DataMid: Data,
        DataOut: Data,
        DataRefIn: Data,
        DataRefMid: Data,
        DataRefOut: Data,
    > OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut>
    for OpBuilderChain<'a, DataIn, DataMid, DataOut, DataRefIn, DataRefMid, DataRefOut>
{
    fn build(
        &mut self,
        sample_data: DataIn,
        sample_ref: DataRefIn,
    ) -> (
        Box<dyn ModelOp<DataIn, DataOut, DataRefIn, DataRefOut>>,
        (DataIn, DataRefIn),
    ) {
        let (mut first_op, (sample_data, sample_ref)) =
            self.first_op.build(sample_data, sample_ref);
        let (sample_data, sample_ref) = first_op.forward_or_transform(sample_data, sample_ref);
        let (second_op, (sample_data, sample_ref)) = self.second_op.build(sample_data, sample_ref);
        let data_and_ref = first_op.backward_or_revert(sample_data, sample_ref);
        (Box::new(OpChain::new(first_op, second_op)), data_and_ref)
    }
}

pub trait CombinatoryOpBuilder<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data> {
    fn push_and_chain<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>;

    fn push_and_pack<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuild<'a, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed>;
}

impl<'a, DataIn: Data, DataOut: Data, DataRefIn: Data, DataRefOut: Data, OpB>
    CombinatoryOpBuilder<'a, DataIn, DataOut, DataRefIn, DataRefOut> for OpB
where
    OpB: OpBuilder<DataIn, DataOut, DataRefIn, DataRefOut> + 'a,
{
    fn push_and_chain<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuilderChain<'a, DataIn, DataOut, DataOutPushed, DataRefIn, DataRefOut, DataRefOutPushed>
    {
        OpBuilderChain::new(Box::new(self), Box::new(op))
    }

    fn push_and_pack<
        DataOutPushed: Data,
        DataRefOutPushed: Data,
        OpBuilderPushed: OpBuilder<DataOut, DataOutPushed, DataRefOut, DataRefOutPushed> + 'a,
    >(
        self,
        op: OpBuilderPushed,
    ) -> OpBuild<'a, DataIn, DataOutPushed, DataRefIn, DataRefOutPushed> {
        OpBuild::new_op_builder(self.push_and_chain(op))
    }
}
