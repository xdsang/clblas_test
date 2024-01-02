/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


__constant float2 twiddles[127] = {
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(9.8078528040323043057924223830923438e-01f, -1.9509032201612824808378832130983938e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654746171500846685376018e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-1.9509032201612819257263709005201235e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654746171500846685376018e-01f),
(float2)(6.1232339957367660358688201472919830e-17f, -1.0000000000000000000000000000000000e+00f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(5.5557023301960228867102387084742077e-01f, -8.3146961230254523567140267914510332e-01f),
(float2)(-3.8268343236508972626808144923415966e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-9.8078528040323043057924223830923438e-01f, -1.9509032201612860890627132448571501e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(-9.2387953251128684950543856757576577e-01f, 3.8268343236508967075693021797633264e-01f),
(float2)(1.9509032201612833135051516819657991e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(-9.2387953251128673848313610506011173e-01f, -3.8268343236508989280153514300764073e-01f),
(float2)(-5.5557023301960217764872140833176672e-01f, 8.3146961230254523567140267914510332e-01f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f),
(float2)(9.9879545620517240500646494183456525e-01f, -4.9067674327418014934565348994510714e-02f),
(float2)(9.9518472667219692873175063141388819e-01f, -9.8017140329560603628777926132897846e-02f),
(float2)(9.8917650996478101443898367506335489e-01f, -1.4673047445536174793190298260014970e-01f),
(float2)(9.9518472667219692873175063141388819e-01f, -9.8017140329560603628777926132897846e-02f),
(float2)(9.8078528040323043057924223830923438e-01f, -1.9509032201612824808378832130983938e-01f),
(float2)(9.5694033573220882438192802510457113e-01f, -2.9028467725446233105301985233381856e-01f),
(float2)(9.8917650996478101443898367506335489e-01f, -1.4673047445536174793190298260014970e-01f),
(float2)(9.5694033573220882438192802510457113e-01f, -2.9028467725446233105301985233381856e-01f),
(float2)(9.0398929312344333819595476597896777e-01f, -4.2755509343028208490977704059332609e-01f),
(float2)(9.8078528040323043057924223830923438e-01f, -1.9509032201612824808378832130983938e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(9.7003125319454397423868385885725729e-01f, -2.4298017990326387094413007616822142e-01f),
(float2)(8.8192126434835504955600526955095120e-01f, -4.7139673682599764203970948983624112e-01f),
(float2)(7.4095112535495910588423384979250841e-01f, -6.7155895484701833009211213720845990e-01f),
(float2)(9.5694033573220882438192802510457113e-01f, -2.9028467725446233105301985233381856e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(6.3439328416364548779426968394545838e-01f, -7.7301045336273699337681364340824075e-01f),
(float2)(9.4154406518302080630888895029784180e-01f, -3.3688985339222005110926261295389850e-01f),
(float2)(7.7301045336273699337681364340824075e-01f, -6.3439328416364548779426968394545838e-01f),
(float2)(5.1410274419322166128409890006878413e-01f, -8.5772861000027211808571792062139139e-01f),
(float2)(9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654746171500846685376018e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(9.0398929312344333819595476597896777e-01f, -4.2755509343028208490977704059332609e-01f),
(float2)(6.3439328416364548779426968394545838e-01f, -7.7301045336273699337681364340824075e-01f),
(float2)(2.4298017990326398196643253868387546e-01f, -9.7003125319454397423868385885725729e-01f),
(float2)(8.8192126434835504955600526955095120e-01f, -4.7139673682599764203970948983624112e-01f),
(float2)(5.5557023301960228867102387084742077e-01f, -8.3146961230254523567140267914510332e-01f),
(float2)(9.8017140329560770162231619906378910e-02f, -9.9518472667219681770944816889823414e-01f),
(float2)(8.5772861000027211808571792062139139e-01f, -5.1410274419322166128409890006878413e-01f),
(float2)(4.7139673682599780857316318360972218e-01f, -8.8192126434835493853370280703529716e-01f),
(float2)(-4.9067674327417785951066520055974252e-02f, -9.9879545620517240500646494183456525e-01f),
(float2)(8.3146961230254523567140267914510332e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-1.9509032201612819257263709005201235e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(8.0320753148064494286728631777805276e-01f, -5.9569930449243335690567846540943719e-01f),
(float2)(2.9028467725446233105301985233381856e-01f, -9.5694033573220893540423048762022518e-01f),
(float2)(-3.3688985339222016213156507546955254e-01f, -9.4154406518302069528658648778218776e-01f),
(float2)(7.7301045336273699337681364340824075e-01f, -6.3439328416364548779426968394545838e-01f),
(float2)(1.9509032201612833135051516819657991e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(-4.7139673682599769755086072109406814e-01f, -8.8192126434835504955600526955095120e-01f),
(float2)(7.4095112535495910588423384979250841e-01f, -6.7155895484701833009211213720845990e-01f),
(float2)(9.8017140329560770162231619906378910e-02f, -9.9518472667219681770944816889823414e-01f),
(float2)(-5.9569930449243291281646861534682103e-01f, -8.0320753148064516491189124280936085e-01f),
(float2)(7.0710678118654757273731092936941423e-01f, -7.0710678118654746171500846685376018e-01f),
(float2)(6.1232339957367660358688201472919830e-17f, -1.0000000000000000000000000000000000e+00f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(6.7155895484701833009211213720845990e-01f, -7.4095112535495910588423384979250841e-01f),
(float2)(-9.8017140329560645262141349576268112e-02f, -9.9518472667219692873175063141388819e-01f),
(float2)(-8.0320753148064505388958878029370680e-01f, -5.9569930449243313486107354037812911e-01f),
(float2)(6.3439328416364548779426968394545838e-01f, -7.7301045336273699337681364340824075e-01f),
(float2)(-1.9509032201612819257263709005201235e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(-8.8192126434835493853370280703529716e-01f, -4.7139673682599786408431441486754920e-01f),
(float2)(5.9569930449243346792798092792509124e-01f, -8.0320753148064483184498385526239872e-01f),
(float2)(-2.9028467725446216451956615856033750e-01f, -9.5694033573220893540423048762022518e-01f),
(float2)(-9.4154406518302069528658648778218776e-01f, -3.3688985339222032866501876924303360e-01f),
(float2)(5.5557023301960228867102387084742077e-01f, -8.3146961230254523567140267914510332e-01f),
(float2)(-3.8268343236508972626808144923415966e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-9.8078528040323043057924223830923438e-01f, -1.9509032201612860890627132448571501e-01f),
(float2)(5.1410274419322166128409890006878413e-01f, -8.5772861000027211808571792062139139e-01f),
(float2)(-4.7139673682599769755086072109406814e-01f, -8.8192126434835504955600526955095120e-01f),
(float2)(-9.9879545620517240500646494183456525e-01f, -4.9067674327417966362308021643912070e-02f),
(float2)(4.7139673682599780857316318360972218e-01f, -8.8192126434835493853370280703529716e-01f),
(float2)(-5.5557023301960195560411648330045864e-01f, -8.3146961230254545771600760417641141e-01f),
(float2)(-9.9518472667219692873175063141388819e-01f, 9.8017140329560145661780268255824922e-02f),
(float2)(4.2755509343028219593207950310898013e-01f, -9.0398929312344333819595476597896777e-01f),
(float2)(-6.3439328416364537677196722142980434e-01f, -7.7301045336273710439911610592389479e-01f),
(float2)(-9.7003125319454397423868385885725729e-01f, 2.4298017990326381543297884491039440e-01f),
(float2)(3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f),
(float2)(-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f),
(float2)(-9.2387953251128684950543856757576577e-01f, 3.8268343236508967075693021797633264e-01f),
(float2)(3.3688985339222005110926261295389850e-01f, -9.4154406518302080630888895029784180e-01f),
(float2)(-7.7301045336273699337681364340824075e-01f, -6.3439328416364548779426968394545838e-01f),
(float2)(-8.5772861000027211808571792062139139e-01f, 5.1410274419322155026179643755313009e-01f),
(float2)(2.9028467725446233105301985233381856e-01f, -9.5694033573220893540423048762022518e-01f),
(float2)(-8.3146961230254534669370514166075736e-01f, -5.5557023301960217764872140833176672e-01f),
(float2)(-7.7301045336273688235451118089258671e-01f, 6.3439328416364559881657214646111242e-01f),
(float2)(2.4298017990326398196643253868387546e-01f, -9.7003125319454397423868385885725729e-01f),
(float2)(-8.8192126434835493853370280703529716e-01f, -4.7139673682599786408431441486754920e-01f),
(float2)(-6.7155895484701866315901952475542203e-01f, 7.4095112535495888383962892476120032e-01f),
(float2)(1.9509032201612833135051516819657991e-01f, -9.8078528040323043057924223830923438e-01f),
(float2)(-9.2387953251128673848313610506011173e-01f, -3.8268343236508989280153514300764073e-01f),
(float2)(-5.5557023301960217764872140833176672e-01f, 8.3146961230254523567140267914510332e-01f),
(float2)(1.4673047445536174793190298260014970e-01f, -9.8917650996478101443898367506335489e-01f),
(float2)(-9.5694033573220882438192802510457113e-01f, -2.9028467725446238656417108359164558e-01f),
(float2)(-4.2755509343028247348783565939811524e-01f, 9.0398929312344311615134984094765969e-01f),
(float2)(9.8017140329560770162231619906378910e-02f, -9.9518472667219681770944816889823414e-01f),
(float2)(-9.8078528040323043057924223830923438e-01f, -1.9509032201612860890627132448571501e-01f),
(float2)(-2.9028467725446327474259078371687792e-01f, 9.5694033573220860233732310007326305e-01f),
(float2)(4.9067674327418125956867811510164756e-02f, -9.9879545620517240500646494183456525e-01f),
(float2)(-9.9518472667219681770944816889823414e-01f, -9.8017140329560825673382851164205931e-02f),
(float2)(-1.4673047445536230304341529517841991e-01f, 9.8917650996478090341668121254770085e-01f),
};


#define fptype float

#define fvect2 float2

#define C8Q  0.70710678118654752440084436210485f

__attribute__((always_inline)) void 
FwdRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}
__attribute__((always_inline)) void 
FwdRad8B1(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + (fvect2)(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * (fvect2)((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + (fvect2)(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * (fvect2)((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

__attribute__((always_inline)) void
FwdPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*1];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 )*1];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 )*1];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 )*1];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 )*1];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*1];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 )*1];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 )*1];
	}

	else
	{
	(*R0) = (fvect2)(0, 0);
	(*R1) = (fvect2)(0, 0);
	(*R2) = (fvect2)(0, 0);
	(*R3) = (fvect2)(0, 0);
	(*R4) = (fvect2)(0, 0);
	(*R5) = (fvect2)(0, 0);
	(*R6) = (fvect2)(0, 0);
	(*R7) = (fvect2)(0, 0);
	}



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7);
	}

}

__attribute__((always_inline)) void
FwdPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float2 *bufIn, __local float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7,__local float2 *twiddles_LMS)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 )*1];


	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 )*1];

	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 )*1];

	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 )*1];
	}

	else
	{
	(*R0) = (fvect2)(0, 0);
	(*R4) = (fvect2)(0, 0);
	(*R1) = (fvect2)(0, 0);
	(*R5) = (fvect2)(0, 0);
	(*R2) = (fvect2)(0, 0);
	(*R6) = (fvect2)(0, 0);
	(*R3) = (fvect2)(0, 0);
	(*R7) = (fvect2)(0, 0);
	}



	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles_LMS[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);



	if(rw)
	{
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7);
	}

}

__attribute__((always_inline)) void
FwdPass2(uint rw, uint b, uint me, uint me_ori, uint inOffset, uint outOffset, __local float2 *bufIn, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7,__local float2 *twiddles_LMS)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 )*1];
	}

	else
	{
	(*R0) = (fvect2)(0, 0);
	(*R4) = (fvect2)(0, 0);
	(*R1) = (fvect2)(0, 0);
	(*R5) = (fvect2)(0, 0);
	(*R2) = (fvect2)(0, 0);
	(*R6) = (fvect2)(0, 0);
	(*R3) = (fvect2)(0, 0);
	(*R7) = (fvect2)(0, 0);
	}



	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 0)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 0)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 0)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 1)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 1)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles_LMS[31 + 3*((2*me + 1)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	barrier(CLK_LOCAL_MEM_FENCE);

	//因为LDS中的下标 以128为一组，组内的下标乘以64就是在DDR中的下标
	//+ me_ori/16是因为看output offset，lds每隔16个local id就跳128，跳了以后，其对应的DDR中的值刚好偏1
	bufOut[ ( (me_ori/16)*128 + 2*me)     %128 *64 + me_ori/16] = (*R0);
	bufOut[ ( (me_ori/16)*128 + 2*me+32 ) %128 *64 + me_ori/16] = (*R1);
	bufOut[ ( (me_ori/16)*128 + 2*me+64 ) %128 *64 + me_ori/16] = (*R2);
	bufOut[ ( (me_ori/16)*128 + 2*me+96 ) %128 *64 + me_ori/16] = (*R3);

	bufOut[ ( (me_ori/16)*128 + 2*me    +1) %128 *64 + me_ori/16] = (*R4);
	bufOut[ ( (me_ori/16)*128 + 2*me+32 +1) %128 *64 + me_ori/16] = (*R5);
	bufOut[ ( (me_ori/16)*128 + 2*me+64 +1) %128 *64 + me_ori/16] = (*R6);
	bufOut[ ( (me_ori/16)*128 + 2*me+96 +1) %128 *64 + me_ori/16] = (*R7);


}

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_fwd(__global const float2 * restrict gbIn, __global float2 * restrict gbOut)
{
	uint me = get_local_id(0);

	uint glbid = get_group_id(0);
	__local float2 lds[1024];
	__local float2 twiddles_LMS[127];
	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	uint rw = 1;

	uint b = 0;
	uint batch,t;

	if (me<127)
	{
		twiddles_LMS[me]=twiddles[me];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (uint loop = 0; loop<8; loop++)
	{
//LMS位置乘以groupID就是在DDR中的位置
//把数组分为了四份给到四个group，每个group有1024个数，再按顺序放大LDS中
//这里不需要管每个group不一样的问题了，因为lwbIn的首地址已经保证了每个group会跳1024
		t = 0;
		batch = glbid*8+loop;//相邻八个group，groupid一样，区别就是loop>1后，加个loop=原来的groupid
		iOffset = (batch/8)*8192 + (batch%8)*1024;
		oOffset = (batch/8)*8192 + (batch%8)*8;
		lwbIn = gbIn + iOffset;
		lwbOut = gbOut + oOffset;

		R0 = lwbIn[ ( (me/16)*128 + me%16    )  ];
		R1 = lwbIn[ ( (me/16)*128 + me%16+16 )  ];
		R2 = lwbIn[ ( (me/16)*128 + me%16+32 )  ];
		R3 = lwbIn[ ( (me/16)*128 + me%16+48 )  ];
		R4 = lwbIn[ ( (me/16)*128 + me%16+64 )  ];
		R5 = lwbIn[ ( (me/16)*128 + me%16+80 )  ];
		R6 = lwbIn[ ( (me/16)*128 + me%16+96 )  ];
		R7 = lwbIn[ ( (me/16)*128 + me%16+112 ) ];

		FwdRad8B1(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
		lds[ (me/16)*128 + me%16*8  + 0] = R0;
		lds[ (me/16)*128 + me%16*8  + 1] = R1;
		lds[ (me/16)*128 + me%16*8  + 2] = R2;
		lds[ (me/16)*128 + me%16*8  + 3] = R3;
		lds[ (me/16)*128 + me%16*8  + 4] = R4;
		lds[ (me/16)*128 + me%16*8  + 5] = R5;
		lds[ (me/16)*128 + me%16*8  + 6] = R6;
		lds[ (me/16)*128 + me%16*8  + 7] = R7;
		barrier(CLK_LOCAL_MEM_FENCE);
		FwdPass1(rw, b, me%16, t*1024 + (me/16)*128, t*1024 + (me/16)*128, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7,twiddles_LMS);
		barrier(CLK_LOCAL_MEM_FENCE);
		FwdPass2(rw, b, me%16,me, t*1024 + (me/16)*128, t*1024 + (me/16)*128, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7,twiddles_LMS);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


