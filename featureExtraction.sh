#!/bin/bash
# $1 caminho para salvar os arquivos
# $2 caminho para o audioAnalysis.py
# $3 caminho para o banco de audios
for k in `seq 1 2`
do
        for j in `seq 1 24`;
        do
        	frase=$k
        	if [ $j -gt 9 ]; then
        		actor=$j
        		echo $actor
        	else
        		actor="0"$j
        		echo $actor
        	fi
        	cd $1
            a="Actor_"$actor
            mkdir $a
            cd $a
            mkdir Frase_1
            mkdir Frase_2
            mkdir Frase_3
            mkdir Frase_4
        	# Formato 00-00-00-00-00-00-00
        	# 1.Tipo (sempre o mesmo (audio))
        	# 2.Speech (sempre o mesmo)
        	# 3.Emoção (01 - 08)
        	# 4.Intensidade da emoção (01 - 02)
        	# 5.Frase (01 - 02)
        	# 6.Repetição (01 - 02)
        	# 7.Ator(01-24)
          cd '..'
          cd '..'
        	for i in `seq 1 8`;
        	do
            echo "valor do i:" $i
	        	a="03-01-0"$i"-01-0"$frase"-01-"$actor".wav"
	        	b=$3"Actor_"$actor"/"$a
	        	c=$1"Actor_"$actor"/Frase_"$frase"/Emotion_0"$i
	        	d=$2" featureExtractionFile -i "$b" -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o "$c
	        	echo "Audio selecionado: " $a \
	        	&& python $d

            echo "valor do i:" $i
            outputRepeticao=$(($frase+2))
            a1="03-01-0"$i"-01-0"$frase"-02-"$actor".wav"
	        	b1=$3"Actor_"$actor"/"$a1
	        	c1=$1"Actor_"$actor"/Frase_"$outputRepeticao"/Emotion_0"$i
	        	d1=$2" featureExtractionFile -i "$b1" -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o "$c1
	        	echo "Audio selecionado: " $b1 \
            && echo $outputRepeticao \
	        	&& python $d1
        	done
        done
done
