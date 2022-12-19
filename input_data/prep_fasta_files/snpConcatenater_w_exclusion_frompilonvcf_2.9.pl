#!/usr/bin/perl -w

#author Maha Farhat

#Reads in all the vcf files in a directory, exclusion BED file, ID failed file and takes options 1) INDEL|SNP and 2) REGION|WHOLE. The first refers to whether to include INDELs, or just SNPs. The second allows for alignment of just a region between two coordinate. If REGION opion provided need to provide START and STOP coordinates and strand.
# The region option results in an MSA of the full region with SNPs/indels introduced, and not just SNP concatenation
# for each line in the vcf file it excludes those lines that have a reference position between the start and end positions of an exclusion BED file
# exports this to a single multiple alignment fasta file with the file name as the first field

##example command: perl snpConcatenater_w_exclusion_frompilonvcf.pl <file_w_coord_start_stop_to_exclude in BED format> <failed_ID_list> [INDEL|SNP] [REGION|WHOLE] <startcoord-zerobased>-<endcoord-1based> <strand:pos or neg> > output_filename.fasta>\n REGION is optional; IF REGION is provided start stop and strand need to both be defined\n"

use warnings;
use strict;

my @tempFiles;

my @fileListRaw =&ReadInFile('/n/data1/hms/dbmi/farhat/tb_cnn/files_20201207.txt');
my @fileList;

#my $l=$#ARGV;
#print STDERR "$l\t@ARGV\n";

if ($#ARGV < 1) {
    print STDERR "example command: perl snpConcatenater_w_exclusion_frompilonvcf.pl <file_w_coord_start_stop_to_exclude in BED format> <failed_ID_list> [INDEL|SNP] REGION <startcoord-zerobased>-<endcoord-1based> <strand:pos or neg> > output_filename.fasta>\n REGION is optional; IF REGION is provided start stop and strand need to both be defined\n";
    die;
}


#########get arguments
my $excludedCoords=shift(@ARGV);
my $failed_IDs=shift(@ARGV);

my $option3=shift(@ARGV);
#print STDERR "$option3\n";

my $option4= shift(@ARGV)|| "";
#print STDERR "$option4\n";

my $regionstart; my $regionend; my $strand;

if ($option4 =~ m/REGION/i ) {
	my $i= shift(@ARGV);
	#print STDERR "$regionstart\n";
	if ($i =~ m/(\d+)-(\d+)/) {
		$regionstart=$1;
		$regionend=$2;
	}
	$strand= shift(@ARGV);
	if ($regionstart !~ /\d+/) {
		print STDERR "If REGION is provided start and stop need to both be defined\nexample command: ./snpConcatenater_w_exclusion_frompilonvcf.pl <file_w_coord_start_stop_to_exclude in BED format> <failed_ID_list> INDEL REGION <startcoord-zerobased>-<endcoord-1based> >output_filename.fasta>\n";
		die;
	}
}

#need a 1-based regionstart from here on
$regionstart++;

my @excludedcoordRaw =&ReadInFile("$excludedCoords");
my @failed_IDsRaw=&ReadInFile("$failed_IDs");
my @Start_excludedcoord;
my @End_excludedcoord;

foreach my $line (@excludedcoordRaw) {
	chomp $line;
	#print STDERR "reading $line\n";
	my @pieces=split /\t/,$line;
	#if ($pieces[1] =~ /^\d+$/) {
	push (@Start_excludedcoord, $pieces[1]);
	push (@End_excludedcoord,$pieces[2]);
	#}
}

foreach my $line (@fileListRaw) {
    chomp $line;
    if (length($line) > 0) {
        $line =~ s/\*//g;
        push (@fileList, $line);
    }
}

my %badIDs;
foreach my $line (@failed_IDsRaw) {
    chomp $line;
    my @pieces=split /\s+/,$line;
    #print STDERR "reading $line...\n";
    $badIDs{$pieces[0]}=1;
}

push(@tempFiles, 'files.txt');

my %masterHash;
my %referenceHash;
my %insertionHash;
my @allStrains;
my @allCoords;
my @insCoords;
my $qualThresh=10;
my $heteroThresh=0.10;
my @RvDNA;


######### get the H37Rv reference sequence for a region based whole sequence alignment i.e not a snp concatenation
if ($option4 =~ m/REGION/i) {
       my @sequence;
       @sequence = `/n/data1/hms/dbmi/farhat/bin/work-horse/bin/get_seq_coord.pl -coord ${regionstart}-${regionend} -nodefline /n/data1/hms/dbmi/farhat/bin/work-horse/bin/h37rv.fasta`; #regionstart here needs to be 1-based
       my $sequence;

       foreach my $line (@sequence) {
            chomp ($line);
            $sequence=$sequence.$line;
       }

       @RvDNA = split( '', $sequence);

       my $r=$regionstart; #already incremented above
       foreach my $base (@RvDNA) {
	    $referenceHash{$r}=$base;
            $r++;
       }
       if ($regionstart>1) {
	       push (@Start_excludedcoord, 1);
       	       push (@End_excludedcoord, $regionstart-1 );
       }
       if ($regionend <4411532) {
	       push (@Start_excludedcoord, $regionend+1);
      	       push (@End_excludedcoord, 4411532);
       }

}

FILE: for (my $i=0; $i<=$#fileList; $i++) {

    my @fileRaw =&ReadInFile("$fileList[$i]"); #("$fileList[$i]".'.out');

    my $strainName = $fileList[$i];
    $strainName =~ s/\.vcf//g;

    if (defined $badIDs{$strainName}) {
		print STDERR "skipping failed $strainName\n";
		next FILE;
    }

    push (@allStrains, $strainName);
    #print STDERR "processing strain $strainName...\n";

    VAR: foreach my $line (@fileRaw) {
       chomp $line;
       #print STDERR $line;
       if ($line =~ /^NC_000962\.3/) {

        	my @elements = split /\t/, $line;
  	        my ($from,$ref_allele,$allele)=($elements[1],@elements[3..4]);
		#print STDERR "$from\n";
		my $ex = &qualityControl($line);

		next VAR if $ex>0;

		if ( length($ref_allele) > 1 || length($allele) > 1) {
                    if (length($ref_allele) == length($allele)) {
				#print STDERR "multi-base substitution found $ref_allele > $allele \n";
				for (my $i=0; $i<length($ref_allele); $i++){
					my $pos=$from+$i;
					my $ref=substr($ref_allele,$i,1);
					my $base=substr($allele,$i,1);
					$masterHash{$strainName}{$pos} = $base;
					$referenceHash{$pos}= $ref;
			        	push(@allCoords, $pos);
				}
                     } elsif ($option3 =~ m/INDEL/i) {
			if (length($ref_allele) ==1 || length($allele) ==1) {
				if (length($ref_allele) > length($allele)) { #deletion
					for (my $i=1; $i<length($ref_allele); $i++){ #ignoring the zero index base in $ref_allele and $allele
						my $pos=$from+$i;
						if ($pos <= $regionend) {
	                        			my $ref=substr($ref_allele,$i,1);
        	       				        $masterHash{$strainName}{$pos} = '-';
							$referenceHash{$pos}= $ref;
                       					push(@allCoords, $pos);
							#print STDERR "storing $strainName $from $ref_allele > $allele as $masterHash{$strainName}{$pos} and $referenceHash{$pos}\n";
						}
					}
				} else { #insertion
					#print STDERR "we have an insertion\n";
					for (my $i=1; $i<length($allele); $i++){ #ignoring the zero index base in $ref_allele and $allele
                                                my $pos=$from+$i;
						my $base=substr($allele, $i, 1);
						$insertionHash{$strainName}{$pos} = $base;
						push(@insCoords, $pos);
					}
				}
			} else {
                		print STDERR "Warning: didnot process complex variant $ref_allele > $allele at genomic coord $from\n";
	        		next VAR;
			}
                   }
                } else { # single base substitution; I think pilon prints each on a separate line

    		 $masterHash{$strainName}{$from} = $allele;

		 #print STDERR "storing $strainName $from $ref_allele > $allele as $masterHash{$strainName}{$from}\n";
		 $referenceHash{$from}= $ref_allele;

		 push(@allCoords, $from);
                }
      	}
   }
}

my @uniqueCoords = &duplicateEliminator(@allCoords);
@uniqueCoords = sort {$a <=> $b} @uniqueCoords;

@insCoords = &duplicateEliminator(@insCoords);
@insCoords = sort {$a <=> $b} @insCoords;

unless (exists($uniqueCoords[0]) || exists($insCoords[0])) {
	die "Warning: no variants in region found! Note: No fasta file written\n";
}

$"="\n";
#print STDERR "Variants found at H37Rv coodinates:\n@uniqueCoords \nInsertions found at:\n@insCoords\n";
$"="";

if ($option4 =~ m/REGION/i) { #expand uniqueCoords to include all sites of the region, and place insertions in right place i.e. not at the end as above code does

	#print STDERR "original gene sequence is @RvDNA\n";
	my @ungappedRv=@RvDNA;
	my $counterXT=0; my $counterIT=0; my $last=0; my $h; #shifts one with the addition of each gap
	foreach my $pos (@insCoords) { #this loop adds gaps where any insertion has occurred relative to H37Rv such there reference RvDNA contains all the gaps
		if ($pos-$last >1) {
			$counterXT=$counterXT+$counterIT;
			$counterIT=1;
		} else {
			$counterIT++;
		}
		$h= $pos-$regionstart+$counterXT-1;
                #print STDERR "insertion at site $pos, counters are external $counterXT internal $counterIT, index is $h\n";
		@RvDNA = (@RvDNA[0..$h],'-',@RvDNA[($h+1)..$#RvDNA]);
		#print STDERR "@RvDNA \n";
		$last=$pos;

	}
	#print STDERR "$counterXT gaps added to the alignment\n sequence is now @RvDNA\n";

	for (my $i=0; $i<=$#allStrains; $i++) {
	    my $m=$allStrains[$i];
	    (my $y = $m) =~ s/-/_/g;
	    if ($y=~ /^\d+/) {
			print ">$y\n";
	    } else {
			print ">$y\n";
            }
            my @sequence= @ungappedRv;
	    foreach my $pos (@uniqueCoords) { #introduces deletions and snps relative to RV
		if (exists $masterHash{$m}{$pos} ) {
			$sequence[$pos-$regionstart]=$masterHash{$m}{$pos};
			#if ($pos>$regionend) {
			#	@RvDNA = (@RvDNA, $referenceHash{$pos});
			#}
		}
	    }
	    #print STDERR "@sequence\n";
            $counterXT=0; $counterIT=0; $last=0; $h=0;
	    foreach my $pos (sort {$a <=> $b} keys %{$insertionHash{$m}}) { #introduces any insertions that strain has
		if ($pos-$last >1) {
                        $counterXT=$counterXT+$counterIT;
                        $counterIT=1;
                } else {
                        $counterIT++;
                }
		$h= $pos-$regionstart+$counterXT-1;
                @sequence = (@sequence[0..$h],$insertionHash{$m}{$pos},@sequence[($h+1)..$#sequence]);
                $last=$pos;
	    }
	    #print STDERR "@sequence\n";
	    $counterXT=0; $counterIT=0; $last=0; $h=0;
	    INS: foreach my $pos (@insCoords) { #introduces gaps into the sequence in place of insertions in other strains
		if ($pos-$last >1) {
			$counterXT=$counterXT+$counterIT;
			$counterIT=1;
		} else {
			$counterIT++;
		}
		if (grep /$pos/, keys %{$insertionHash{$m}}) {
			#print STDERR "strain $m and $pos in insertion hash\n";
		} else {
			$h=$pos-$regionstart+$counterXT-1;
			@sequence = (@sequence[0..$h],'-',@sequence[($h+1)..$#sequence]);
		}
		$last=$pos;
	    }
            #print STDERR "@sequence\n";
            if ($strand =~ m/neg/i) {
		@sequence = &revComp(@sequence);
            }
	    $"="";
            #print STDERR "length of the sequence is $#sequence \n";
            print "@sequence\n";
	}
	print ">MT_H37Rv\n";
	if ($strand =~ m/neg/i) {
			@RvDNA = &revComp(@RvDNA);
	}
	print STDERR "length of the Rv sequence is $#RvDNA \n";
	print "@RvDNA\n";

} else { #ALL GENOME VARIANTS INCLUDED ONLY SNPCONCATENATE ALIGNMENTS ARE PRODUCED

        for (my $j=0; $j<=$#uniqueCoords; $j++) {
		 push(@RvDNA, $referenceHash{$uniqueCoords[$j]});
	}
        my $counterXT=0; my $counterIT=0; my $last=0; my $h; #shifts one with the addition of each gap
        foreach my $pos (@insCoords) { #this loop adds gaps where any insertion has occurred relative to H37Rv such there reference RvDNA contains all the gaps
                if ($pos-$last >1) {
                        $counterXT=$counterXT+$counterIT;
                        $counterIT=1;
                } else {
                        $counterIT++;
                }
		POSLOOP: for (my $j=0; $j<=$#uniqueCoords; $j++) {
                                if ($uniqueCoords[$j]>=($pos-1)) {
                                       	$h=$j;
                                       	last POSLOOP;
                               	}
                }
		if ($uniqueCoords[$#uniqueCoords]<$pos) { #need this in case insertions happen at the end of the alignment
			@RvDNA=(@RvDNA,'-');
		} elsif ($uniqueCoords[0] >$pos) {
			@RvDNA=('-',@RvDNA);
		} else {
	                $h= $h+$counterXT;
        	        #print STDERR "insertion at site $pos, counters are external $counterXT internal $counterIT, index is $h\n";
                	@RvDNA = (@RvDNA[0..$h],'-',@RvDNA[($h+1)..$#RvDNA]);
	                #print STDERR "@RvDNA \n";
		}
                $last=$pos;

        }

	for (my $i=0; $i<=$#allStrains; $i++) {
	    my $m=$allStrains[$i];
	    $m=~ s/-/_/g;
	    if ($m=~ /^\d+/) {
			print ">$m\n";
	    } else {
			print ">$m\n";
	    }
            my @sequence;
	    for (my $j=0; $j<=$#uniqueCoords; $j++) {
		#print STDERR "coord position $uniqueCoords[$j]\n";
	        if (exists $masterHash{$allStrains[$i]}{$uniqueCoords[$j]}) {
			#print STDERR "here!\n";
			push(@sequence,$masterHash{$allStrains[$i]}{$uniqueCoords[$j]});
		} else {
			#print STDERR "and here!\n";
			push(@sequence,$referenceHash{$uniqueCoords[$j]});
		}
	    }
	    #print STDERR "@sequence\n";
	    $counterXT=0; $counterIT=0; $last=0; $h=0;
            foreach my $pos (@insCoords) { #introduces insertions and gaps into the sequence in place of insertions in	other strains
              if ($pos-$last >1) {
                       $counterXT=$counterXT+$counterIT;
                       $counterIT=1;
                       POSLOOP: for (my $j=0; $j<=$#uniqueCoords; $j++) {
                               if ($uniqueCoords[$j]>=($pos-1)) {
                                       $h=$j;
                                     	last POSLOOP;
                              	}
                       }
               } else {
                       $counterIT++;
		       $h++;
               }
               if (grep /^$pos$/, keys %{$insertionHash{$m}}) {
			#print STDERR "found one $pos!\n";
		        if ($uniqueCoords[$#uniqueCoords]<$pos) { #need this in case insertions happen at the end of the alignment
        	                @sequence=(@sequence,$insertionHash{$m}{$pos});
			} elsif ($uniqueCoords[0] >$pos) {
                  		@sequence=($insertionHash{$m}{$pos},@sequence);
 	        	} else {
				my $i= $h+$counterXT;
	                        #print STDERR "my index now is $h\n";
        	                @sequence = (@sequence[0..$i],$insertionHash{$m}{$pos},@sequence[($i+1)..$#sequence]);
                	        #print STDERR "@sequence\n";
                	}

                } else {
	                if ($uniqueCoords[$#uniqueCoords]<$pos) { #need this in case insertions happen at the end of the alignment
        	                @sequence=(@sequence,'-');
			} elsif ($uniqueCoords[0] >$pos) {
                        	@sequence=('-',@sequence);
               		} else {
  				my $i=$h+$counterXT;
        	                @sequence = (@sequence[0..$i],'-',@sequence[($i+1)..$#sequence]);
			}
                }
		$last=$pos;
             }
             $"="";
	     #my $length=scalar(@sequence);
	     #print STDERR "length of the sequence is $length\n";
             print "@sequence\n";
	 }
        #my $lenght=scalar(@RvDNA);
	#print STDERR "length of the Rv sequence is $lenght\n";
	print ">MT_H37Rv\n@RvDNA\n";
}



# foreach my $rmFile (@tempFiles) { system('rm '.$rmFile); }

########################################################################################
#SUBROUTINES----SUBROUTINES----SUBROUTINES----SUBROUTINES----SUBROUTINES----SUBROUTINES#
########################################################################################

#---------------------------------------------------------------------------------------
# Reads in a file and stores all the elements to an array
#---------------------------------------------------------------------------------------
sub ReadInFile
{
    my @FileName = @_;
    my @FileContents;
    open (FILE_OF_INTEREST, $FileName[0]) or die ("Unable to open the file called $FileName[0]");
    @FileContents = <FILE_OF_INTEREST>;
    close FILE_OF_INTEREST;
    return @FileContents;
}

#---------------------------------------------------------------------------------------
# Writes out a file
#---------------------------------------------------------------------------------------
sub WriteOutFile
{
  my @fileName = shift @_;
  my @fileContents =  @_;
  open (FILE_TO_WRITE_OUT, ">$fileName[0]") or die ("Unable to open the file called $fileName[0]");
  print FILE_TO_WRITE_OUT "@fileContents";
  close FILE_TO_WRITE_OUT;
}

#---------------------------------------------------------------------------------------
# Checks for duplications
#---------------------------------------------------------------------------------------
sub duplicateEliminator
{
    my @duplicateList = @_;
    my %uniqueHash;
    foreach my $i (@duplicateList) { $uniqueHash{$i} = 0;}
    my @uniqueList = keys(%uniqueHash);
    return @uniqueList;
}

#---------------------------------------------------------------------------------------
#  Check the qualtiy of a variant
#---------------------------------------------------------------------------------------

sub qualityControl
{
	my $line = shift @_;
	my $ex=0;
	my @elements = split /\t/, $line;
	my ($from,$ref_allele,$allele,$snpqual,$filter,$info)=($elements[1],@elements[3..7]);
	$ex=1 if $info =~ /IMPRECISE/i;
	(my $depth) = ($info =~ /DP=(\d+)/);
	#(my $tcf)   = ($info =~ /TCF=(\d+)/);
	#(my $tcr)   = ($info =~ /TCR=(\d+)/);
	#(my $nf)    = ($info =~ /NF=(\d+)/);
	#(my $nr)    = ($info =~ /NR=(\d+)/);
	#my ($dpr1,$dpr2,$dp1,$dp2) = ($tcf-$nf, $tcr-$nr, $nf, $nr);
	#my $bidir = ($dp1 && $dp2)?'Y':'N';
	#(my $A) = ($info =~ /BC=(\d+),/);
	#(my $C) = ($info =~ /BC=\d+,(\d+),/);
	#(my $G) = ($info =~ /BC=\d+,\d+,(\d+),/);
	#(my $T)	= ($info =~ /BC=\d+,\d+,\d+,(\d+)/);
	$ex=2 if $ref_allele =~ /N/; #ambiguous reference
	$ex=2 if $allele =~ /N/; #ambiguous/imprecise change
	$ex=3 if $allele =~ m/,|</; #heterogenous allele or <dup>
  	my $hqr;
	if ($info =~ /AF=/) {
        	($hqr)= ($info =~ /AF=([0-9|\.]*)/);
  	} else { #imprecise excluded above
        	#print STDERR "here!!!!\n";
        	$hqr=0.766;
  	}
	#print STDERR "$info\n$hqr\n";
	if ($snpqual eq ".") {
	    $snpqual=11;
  	}
	if ($filter =~ m/;/ || $snpqual <$qualThresh ) {
      	    $ex=1;
  	}
  	if ($filter =~ m/PASS|AMB/i && $hqr >= $heteroThresh ) {
  	} else {
      		$ex=1;
  	}
        INTER: for (my $i=$#Start_excludedcoord; $i>=0; $i=$i-1) {
		#print STDERR "$Start_excludedcoord[$i] $End_excludedcoord[$i]\n";
    		if (($from >= $Start_excludedcoord[$i]) && ($from <= $End_excludedcoord[$i])) {
			$ex=5;
			last INTER;
        	}
    	}
    	return $ex;
}

#---------------------------------------------------------------------------------------
#  Check the qualtiy of a variant
#---------------------------------------------------------------------------------------
sub revComp
{
	my $reversecomplement="";
	my @d= @_;
	foreach my $nucleotide(reverse(@d)) {
		if ($nucleotide =~ /a/i) {
			$reversecomplement.="T";
		} elsif ($nucleotide =~ /t/i) {
			$reversecomplement.="A";
		} elsif ($nucleotide =~ /g/i) {
			$reversecomplement.="C";
		} elsif ($nucleotide =~ /c/i) {
			$reversecomplement.="G";
		} elsif ($nucleotide =~ /-/) {
			$reversecomplement.="-";
		} else {
			die "$0:  Bad nucleotide!  [$nucleotide]\n";
		}
	}
	return split( '', $reversecomplement);
}
