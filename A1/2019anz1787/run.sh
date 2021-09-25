part=$2
shift 2

# creating list of arguments
args=""
for ITEM in "$@"
do
    args="$args $ITEM" 
done

python part${part}.py $args
