user_name=$1
script_path="$( cd -- "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"

filename='scripts/script_base.txt'
mkdir -p -- "commands"
filename_output='commands/script_'$user_name'.txt'

echo "Script for user: "$user_name > $filename_output

n=1
while read line; do
echo $line | sed -e "s/{user_name}/${user_name}/g" | sed -e "s%{script_path}%${script_path}%g"  >> $filename_output
n=$((n+1))
done < $filename